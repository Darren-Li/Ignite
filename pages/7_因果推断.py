import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from econml.dml import CausalForestDML
from services.data_loader import load_data, var_profiling

# =========================================================
# Page Config
# =========================================================
st.set_page_config(layout="wide")
st.title("Causal Inference")

# =========================================================
# Session State
# =========================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "features" not in st.session_state:
    st.session_state.features = []

# =========================================================
# Sidebar
# =========================================================
step = st.sidebar.radio(
    "Workflow",
    [
        "Step 0 – Data & Variables",
        "Step 1 – Preprocessing",
        "Step 2 – Variable Selection",
        "Step 3 – Modeling",
        "Step 4 – Evaluation",
        "Step 5 – CRM Decision Panel",
        "Step 6 – Scoring"
    ]
)

# =========================================================
# Step 0 - Variable Config
# =========================================================
if step == "Step 0 – Data & Variables":

    st.header('Step 0: Data Import & Variable Classification')
    # =================    Load data    =================
    st.subheader('1. Load Data')

    df = load_data(key="Causal Forest")
    st.session_state.df = df

    # ================= Variable config =================
    st.subheader('2. Variable Profiling & Classification')

    config = var_profiling(df, 
        options = ['exclude', 'id', 'target', 'treatment', 'binary', 'nominal', 'ordinal', 'continuous']
    )

    if st.button("💾 Save Config"):
        st.session_state.var_config = config
        # os.makedirs("config", exist_ok=True)
        # yaml.safe_dump(config, open("config/uplift_vars.yaml", "w"))
        st.success("Saved")

# =========================================================
# Step 1 - Preprocessing
# =========================================================
if step == "Step 1 – Preprocessing" and st.session_state.df is not None:

    st.header("Step 1: Preprocessing")

    df = st.session_state.df.copy()
    config = st.session_state.var_config

    id_var = [k for k,v in config.items() if v == "id"][0]
    treatment = [k for k,v in config.items() if v=="treatment"][0]
    target = [k for k,v in config.items() if v=="target"][0]

    features = [k for k,v in config.items() if v not in ["exclude","id","target","treatment"]]

    df = df[[treatment, target] + features]

    num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if len(cat_cols)>0:
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

    st.session_state.processed_df = df
    st.session_state.features = features

    st.success("Preprocessing done")
    st.dataframe(df.head())

    os.makedirs("config", exist_ok=True)

    artifact = {
        "features": features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoder": enc
    }

    joblib.dump(artifact, "config/uplift_artifact.pkl")

    # 额外再存一个yaml（用于可读性/追踪）
    yaml_meta = {
        "id_var": id_var,
        "treatment_var":treatment,
        "target_var": target,
        "features": features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    with open("config/uplift_preprocess.yaml", "w") as f:
        yaml.safe_dump(yaml_meta, f)


# =========================================================
# Step 2 - Modeling
# =========================================================
if step == "Step 3 – Modeling":

    st.header("Step 3: Causal Forest Training")

    df = st.session_state.processed_df
    config = st.session_state.var_config

    treatment = [k for k,v in config.items() if v=="treatment"][0]
    target = [k for k,v in config.items() if v=="target"][0]
    features = st.session_state.features

    if st.button("🚀 Train"):

        X = df[features]
        T = df[treatment]
        Y = df[target]

        model = CausalForestDML(
            model_t=RandomForestClassifier(),
            model_y=RandomForestClassifier(),
            discrete_treatment=True,
            discrete_outcome=True,
            n_estimators=300,
            min_samples_leaf=10,
            random_state=42
        )

        model.fit(Y, T, X=X)

        st.session_state.model = model

        artifact = joblib.load("config/uplift_artifact.pkl")
        artifact["model"] = model
        joblib.dump(artifact, "config/uplift_artifact.pkl")

        st.success("Model trained")

# =========================================================
# Step 4 - Evaluation
# =========================================================
if step == "Step 4 – Evaluation":

    st.header("Step 4: Model Evaluation")

    model = st.session_state.model
    df = st.session_state.processed_df
    config = st.session_state.var_config

    treatment_col = [k for k, v in config.items() if v == "treatment"][0]
    target_col = [k for k, v in config.items() if v == "target"][0]
    features = st.session_state.features

    # =========================================================
    # 1️⃣ Predict Uplift
    # =========================================================
    uplift = model.effect(df[features])
    df_eval = df.copy()
    df_eval["uplift"] = uplift

    st.subheader("1. Uplift Distribution")

    col1, col2 = st.columns([4,1])

    with col1:
        st.write("Uplift Histogram")

        bins = pd.cut(df_eval["uplift"], 20)

        hist = bins.value_counts().sort_index()

        centers = [b.mid for b in hist.index]

        hist_df = pd.DataFrame({
            "uplift_bin_center": centers,
            "count": hist.values
        })

        st.bar_chart(hist_df.set_index("uplift_bin_center"))

    with col2:
        st.dataframe(df_eval[["uplift"]].describe())

    st.divider()

    col1, col, col2 = st.columns([3, 1/2, 2])

    with col1:

        # =========================================================
        # 2️⃣ Qini Curve (核心🔥)
        # =========================================================
        st.subheader("2. Qini Curve (Model Quality)")

        def qini_curve(df, treatment, outcome, uplift):

            df = df.copy()
            df = df.sort_values(uplift, ascending=False).reset_index(drop=True)

            df["treatment"] = df[treatment]
            df["outcome"] = df[outcome]

            # cumulative sums
            df["cum_treat"] = df["treatment"].cumsum()
            df["cum_control"] = (1 - df["treatment"]).cumsum()

            df["cum_y_treat"] = (df["outcome"] * df["treatment"]).cumsum()
            df["cum_y_control"] = (df["outcome"] * (1 - df["treatment"])).cumsum()

            # avoid divide by zero
            df["treat_rate"] = df["cum_y_treat"] / (df["cum_treat"] + 1e-6)
            df["control_rate"] = df["cum_y_control"] / (df["cum_control"] + 1e-6)

            df["qini"] = df["treat_rate"] - df["control_rate"]

            return df["qini"]

        qini = qini_curve(df_eval, treatment_col, target_col, "uplift")

        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(qini.values)
        ax.set_title("Qini Curve (Uplift Ranking Power)")
        ax.set_xlabel("Population (ranked by uplift)")
        ax.set_ylabel("Qini Gain")

        st.pyplot(fig, dpi=150)

    with col2:
        # =========================================================
        # 3️⃣ AUUC (核心指标🔥)
        # =========================================================
        st.subheader("3. AUUC (Area Under Uplift Curve)")

        auuc = np.trapz(qini.values)

        st.metric("AUUC", f"{auuc:.4f}")
        st.metric("Max Qini", f"{qini.max():.4f}")
        st.metric("Min Qini", f"{qini.min():.4f}")

        st.caption("AUUC 越大说明 uplift 排序能力越强（核心指标）")

    st.divider()

    # =========================================================
    # 4️⃣ Uplift@K（业务最重要🔥）
    # =========================================================
    st.subheader("4. Uplift @ K (Business KPI)")

    k = st.slider("Top K % Users", 5, 50, 20)

    threshold = np.percentile(df_eval["uplift"], 100 - k)

    top_df = df_eval[df_eval["uplift"] >= threshold]

    treated = top_df[top_df[treatment_col] == 1][target_col].mean()
    control = top_df[top_df[treatment_col] == 0][target_col].mean()

    uplift_k = treated - control

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Selected Users", len(top_df))
    col2.metric("Treatment Conv", f"{treated:.4f}")
    col3.metric("Control Conv", f"{control:.4f}")
    col4.metric("Uplift@K", f"{uplift_k:.4f}")

    st.caption("衡量：只在 top uplift 人群里，干预是否真的有效")

    st.divider()

    # =========================================================
    # 5️⃣ Uplift Segmentation（可解释性🔥）
    # =========================================================
    st.subheader("5. Uplift Segments (Actionable Groups)")

    df_eval["segment"] = pd.qcut(df_eval["uplift"], 5, labels=[
        "Very Low", "Low", "Medium", "High", "Very High"
    ])

    seg_table = df_eval.groupby("segment").agg(
        count=(target_col, "count"),
        conversion_rate=(target_col, "mean"),
        avg_uplift=("uplift", "mean")
    ).reset_index()

    col1, col, col2 = st.columns([3,1/2,2])
    col2.dataframe(seg_table)

    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(seg_table["segment"], seg_table["conversion_rate"])
    ax.set_title("Conversion Rate by Uplift Segment")
    col1.pyplot(fig, dpi=150)

# =========================================================
# Step 5 - CRM Decision Panel
# =========================================================
if step == "Step 5 – CRM Decision Panel":

    st.header("💡 CRM Decision Panel")

    df = st.session_state.processed_df
    model = st.session_state.model
    config = st.session_state.var_config

    target = [k for k,v in config.items() if v=="target"][0]
    treatment = [k for k,v in config.items() if v=="treatment"][0]

    features = st.session_state.features

    # =========================================================
    # 1️⃣ Predict uplift
    # =========================================================
    uplift = model.effect(df[features])
    df = df.copy()
    df["uplift"] = uplift

    # =========================================================
    # 2️⃣ Target selection
    # =========================================================
    top_k = st.slider("Top % Target Users", 5, 50, 20)

    threshold = np.percentile(df["uplift"], 100 - top_k)
    df["target_flag"] = (df["uplift"] >= threshold).astype(int)

    target_df = df[df["target_flag"] == 1]
    control_df = df[df["target_flag"] == 0]

    # =========================================================
    # 3️⃣ TRUE UPLIFT CALCULATION（关键🔥）
    # =========================================================
    def calc_uplift(data):
        treated = data[data[treatment] == 1][target].mean()
        control = data[data[treatment] == 0][target].mean()
        return treated - control

    target_uplift = calc_uplift(target_df)
    base_uplift = calc_uplift(df)

    # =========================================================
    # 4️⃣ BUSINESS KPI
    # =========================================================
    col1, col, col2 = st.columns([2,1/2,2])
    cost_per_user = col1.number_input("Cost per user", value=10.0)
    revenue_per_conv = col2.number_input("Revenue per conversion", value=100.0)

    expected_gain = len(target_df) * target_uplift * revenue_per_conv
    total_cost = len(target_df) * cost_per_user
    roi = expected_gain / total_cost if total_cost > 0 else 0

    # =========================================================
    # 5️⃣ OUTPUT
    # =========================================================
    col1, col2, col3 = st.columns(3)

    col1.metric("Target Users", len(target_df))
    col2.metric("Uplift (Target)", f"{target_uplift:.4f}")
    col3.metric("ROI", f"{roi:.2f}")

    st.divider()

    # =========================================================
    # 6️⃣ Comparison view
    # =========================================================
    st.subheader("📊 Strategy Comparison")

    comp_df = pd.DataFrame({
        "Group": ["All Users", "Targeted Users"],
        "Uplift": [base_uplift, target_uplift]
    })

    st.bar_chart(comp_df.set_index("Group"))

# =========================================================
# Step 6 - Scoring
# =========================================================
if step == "Step 6 – Scoring":

    st.header("Step 6: Scoring")

    artifact = joblib.load("config/uplift_artifact.pkl")

    file = st.file_uploader("Upload scoring CSV")

    if file:
        new_df = pd.read_csv(file)

        Xnew = new_df.reindex(columns=artifact["features"])

        Xnew[artifact["num_cols"]] = artifact["num_imputer"].transform(Xnew[artifact["num_cols"]])
        Xnew[artifact["cat_cols"]] = artifact["cat_imputer"].transform(Xnew[artifact["cat_cols"]])
        Xnew[artifact["cat_cols"]] = artifact["encoder"].transform(Xnew[artifact["cat_cols"]].astype(str))

        uplift = artifact["model"].effect(Xnew)

        new_df["uplift_score"] = uplift

        st.dataframe(new_df.head())

        st.download_button(
            "Download",
            new_df.to_csv(index=False).encode(),
            "uplift_scoring.csv"
        )