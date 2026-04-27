import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml
import joblib

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from services.db_service import get_conn

# =========================================================
# Page Config
# =========================================================
st.set_page_config(layout="wide")
st.title("Survival Analysis Engine")

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
# Sidebar Workflow
# =========================================================
step = st.sidebar.radio(
    "Workflow",
    [
        "Step 0 – Data & Variables",
        "Step 1 – Preprocessing",
        "Step 2 – Variable Selection",
        "Step 3 – Modeling",
        "Step 4 – Evaluation",
        'Step 5 – CRM Decision Panel',
        "Step 6 – Scoring",
    ]
)

# =========================================================
# Step 0 - Data Load & Variable Setup
# =========================================================
if step == "Step 0 – Data & Variables":

    st.header('Step 0: Data Import & Variable Classification')
    st.subheader('1. Load Data')

    conn = get_conn()
    sources = pd.read_sql("SELECT * FROM data_sources", conn)

    with st.form("load_form"):
        src = st.selectbox("Select a dataset for modeling", sources["name"], index=None, placeholder="Select a dataset...")
        submit = st.form_submit_button("▶️ Data Preparation & Analysis")

    if submit:
        path = sources.loc[sources["name"] == src, "path"].values[0]
        path = path.replace("\\", "/")

        df = pd.read_csv(path)
        st.session_state.df = df

    if st.session_state.df is None:
        st.info("Please load dataset")
        st.stop()

    df = st.session_state.df
    st.success(f"Data loaded: **{df.shape[0]}** rows, **{df.shape[1]}** columns")

    # ================= Variable config =================
    st.subheader('2. Variable Profiling & Classification')

    sample_n = 3
    col_widths = [1/2, 2, 1, 1*sample_n, 1, 1, 2]
    header_cols = st.columns(col_widths)
    header_texts = ["#", "Field Name", "Type", f"Sample ({sample_n})", "Missing %", "Unique", "Classification"]

    for col, text in zip(header_cols, header_texts):
        col.markdown(f"**{text}**")

    st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)  # 表头分隔线

    prev_config = st.session_state.get('var_config', {})
    options = ['exclude', 'id', 'target_time', 'target_event', 'binary', 'nominal', 'ordinal', 'continuous']

    config = {}
    for idx, c in enumerate(df.columns, 1):
        c_type = df[c].dtype
        c_unique = df[c].nunique()
        cols = st.columns(col_widths)
        # ====== 填充列内容 ======
        with cols[0]:
            st.write(idx)
        with cols[1]:
            st.write(c)
        with cols[2]:
            st.write(str(c_type))
        with cols[3]:
            samples = [str(x) for x in df[c].dropna().unique()[:sample_n]]
            st.markdown(", ".join(samples), unsafe_allow_html=True)
        with cols[4]:
            missing_pct = df[c].isna().mean() * 100
            st.write(f"{missing_pct:.1f}%")
        with cols[5]:
            st.write(c_unique)

        # ===== 自动填充下拉框 =====
        # 先尝试从已有配置取值
        default_value = prev_config.get(c, None)

        # 如果没有，就根据简单规则判断
        if default_value is None:
            if 'id' == c.lower() or '_id' in c.lower():
                default_value = 'id'
            elif 'time' == c.lower() or '_time' in c.lower():
                default_value = 'target_time'
            elif 'event' in c.lower() or '_event' in c.lower():
                default_value = 'target_event'
            elif c_unique == 2:
                default_value = 'binary'
            elif c_type == 'object':
                default_value = 'nominal'
            elif np.issubdtype(c_type, np.number):
                default_value = 'continuous'
            else:
                default_value = 'exclude'
        # 计算默认索引
        default_index = options.index(default_value) if default_value in options else 0

        with cols[6]:
            config[c] = st.selectbox(
                'select the processing method for the feature or the type of feature',
                options,
                index=default_index,
                key=f'var_{c}_{idx}',
                label_visibility="collapsed"
            )

        # 分隔线
        st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)

    if st.button("💾 Save Variable Configuration"):
        st.session_state.var_config = config
        os.makedirs("config", exist_ok=True)
        with open("config/survival_vars.yaml", "w") as f:
            yaml.safe_dump(config, f)
        st.success("Variable configuration saved")

# =========================================================
# Step 1 - Preprocessing
# =========================================================
if step == "Step 1 – Preprocessing" and st.session_state.df is not None:

    st.header("Step 1: Preprocessing")

    df = st.session_state.df.copy()
    config = st.session_state.var_config

    id_var = [k for k,v in config.items() if v == "id"][0]
    time_var = [k for k,v in config.items() if v == "target_time"][0]
    event_var = [k for k,v in config.items() if v == "target_event"][0]

    features = [k for k,v in config.items() if v not in ["exclude","id","target_time","target_event"]]

    df = df[[time_var, event_var] + features]

    # 自动类型
    num_cols = df[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    # missing
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    # encoding
    enc = None
    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))

    st.success("Preprocessing done")

    st.session_state.processed_df = df
    st.session_state.features = num_cols + cat_cols
    
    st.write('The preprocessed dataset')
    st.dataframe(df.head())

    os.makedirs("config", exist_ok=True)

    # 保存特征处理方法，便于scoring时用到
    artifact = {
        "id_var": id_var,
        "time_var": time_var,
        "event_var": event_var,
        "features": st.session_state.features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "ordinal_encoder": enc if len(cat_cols) > 0 else None,
        "num_imputer": SimpleImputer(strategy="median").fit(df[num_cols]) if len(num_cols) > 0 else None,
        "cat_imputer": SimpleImputer(strategy="most_frequent").fit(df[cat_cols]) if len(cat_cols) > 0 else None
    }

    joblib.dump(artifact, "config/survival_preprocess_artifact.pkl")

    # 额外再存一个yaml（用于可读性/追踪）
    yaml_meta = {
        "id_var": id_var,
        "time_var": time_var,
        "event_var": event_var,
        "features": st.session_state.features,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

    with open("config/survival_preprocess.yaml", "w") as f:
        yaml.safe_dump(yaml_meta, f)

# =========================================================
# Step 2 - Variable Selection
# =========================================================
if step == "Step 2 – Variable Selection" and st.session_state.processed_df is not None:

    st.header("Step 2: Variable Selection")

    df = st.session_state.processed_df

    howmany = st.slider("Keep features", 5, min(50, df.shape[1]), 20)

    features = st.session_state.features
    selected = df[features].var().sort_values(ascending=False).head(howmany).index.tolist()

    st.session_state.features = selected

    st.write('Selected variables')
    st.write(selected)

    # =========================================================
    # FINALIZE ARTIFACT AFTER FEATURE SELECTION + TRAINING
    # =========================================================
    artifact = joblib.load("config/survival_preprocess_artifact.pkl")
    artifact["features"] = st.session_state.features  # ⭐最终特征
    joblib.dump(artifact, "config/survival_preprocess_artifact.pkl")

# =========================================================
# Step 3 - Modeling (RSF)
# =========================================================
if step == "Step 3 – Modeling" and st.session_state.processed_df is not None:

    st.header("Step 3: Random Survival Forest")

    df = st.session_state.processed_df
    features = st.session_state.features
    config = st.session_state.var_config

    time_var = [k for k,v in config.items() if v == "target_time"][0]
    event_var = [k for k,v in config.items() if v == "target_event"][0]

    test_size = st.slider("Test size", 0.1, 0.5, 0.3)

    if st.button("🚀 Train Model"):

        X = df[features]
        y = Surv.from_dataframe(event_var, time_var, df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.success("The model training was successful.")

        # =========================================================
        # FINALIZE ARTIFACT AFTER MODEL TRAINING
        # =========================================================
        artifact = joblib.load("config/survival_preprocess_artifact.pkl")
        artifact["model"] = st.session_state.model
        joblib.dump(artifact, "config/survival_preprocess_artifact.pkl")

# =========================================================
# Step 4 - SURVIVAL MODEL EVALUATION (RSF)
# =========================================================
if step == "Step 4 – Evaluation" and st.session_state.model is not None:

    import matplotlib.pyplot as plt

    st.header("Step 4: Model Evaluation")

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # =========================================================
    # 1. Risk Score + C-index
    # =========================================================
    st.subheader("Key KPIs")

    risk = model.predict(X_test)

    cindex = concordance_index_censored(
        y_test["event"],
        y_test["time"],
        risk
    )[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("C-index", f"{cindex:.4f}", 
        help="衡量模型预测的“风险排序”是否正确；回答“❓ 是否“更早流失的人”被模型打了更高风险分？”；建议>0.6")
    col2.metric("Mean Risk", f"{np.mean(risk):.4f}", help="平均“相对风险分数”，越大越可能早流失")
    col3.metric("Max Risk", f"{np.max(risk):.4f}")

    st.divider()

    # =========================================================
    # 2. Risk Distribution + Event Rate by Bin (FIXED)
    # =========================================================
    st.subheader("Risk Segmentation (Event Rate by Risk Decile)")

    risk_df = pd.DataFrame({
        "risk": risk,
        "event": y_test["event"],
        "time": y_test["time"]
    })

    risk_df["risk_bin"] = pd.qcut(
        risk_df["risk"],
        10,
        labels=[f"D{i+1}" for i in range(10)]
    )

    lift_df = (
        risk_df.groupby("risk_bin", observed=False)
        .agg(
            event_rate=("event", "mean"),
            count=("event", "count")
        )
        .reset_index()
    )
    cols = st.columns([3,2])
    with cols[0]:
        st.bar_chart(lift_df.set_index("risk_bin")["event_rate"])
    with cols[1]:
        st.dataframe(lift_df)

    st.divider()

    cols = st.columns(2)
    with cols[0]:
        # =========================================================
        # 3. Survival Curve (Single Sample)
        # =========================================================
        st.subheader("Individual Survival Curve")

        idx = st.slider("Select sample index", 0, len(X_test)-1, 0, help="选取一个数据ID")

        surv_fn = model.predict_survival_function(X_test.iloc[[idx]])[0]

        curve_df = pd.DataFrame({
            "time": surv_fn.x,
            "survival_prob": surv_fn.y
        })

        st.caption(f"用户 **{idx}** 的“未来存活概率曲线”，表示其在未来某个时间点仍然“未流失”的概率")
        st.line_chart(curve_df.set_index("time"))

    with cols[1]:
        # =========================================================
        # 4. Risk Group Survival Curves (High/Mid/Low)
        # =========================================================
        st.subheader("Survival by Risk Groups")
        st.caption("不同风险人群的“生存下降速度”")

        df_eval = risk_df.copy()

        df_eval["risk_group"] = pd.qcut(
            df_eval["risk"],
            3,
            labels=["Low Risk", "Mid Risk", "High Risk"]
        )

        fig, ax = plt.subplots(figsize=(9,5))

        for g in ["Low Risk", "Mid Risk", "High Risk"]:

            sub = df_eval[df_eval["risk_group"] == g]

            if len(sub) < 10:
                continue

            # empirical survival curve
            times = np.sort(sub["time"].unique())
            surv = [(sub["time"] >= t).mean() for t in times]

            ax.plot(times, surv, label=g)

        ax.set_title("Empirical Survival by Risk Group")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.legend()

        st.pyplot(fig, dpi=150)

    st.divider()

    # =========================================================
    # 5. Business KPI: Churn @ Time Horizon
    # =========================================================
    st.subheader("Business KPI (Churn at Time Horizon)")

    max_t = int(df_eval["time"].max())

    t = st.slider("Churn horizon (time)", 1, max_t, min(30, max_t))

    df_eval["churn_t"] = (
        (df_eval["time"] <= t) & (df_eval["event"] == 1)
    ).astype(int)

    churn_table = (
        df_eval.groupby("risk_bin", observed=False)["churn_t"]
        .mean()
        .reset_index()
        .sort_values("risk_bin")
    )

    cols = st.columns([3,2])
    with cols[0]:
        # Lift calculation (business value)
        top = churn_table["churn_t"].iloc[-1]
        bottom = churn_table["churn_t"].iloc[0] + 1e-6
        st.metric("Top vs Bottom Lift", f"{top / bottom:.2f}x", help="高风险用户 vs 低风险用户，差多少倍流失？")

        st.bar_chart(churn_table.set_index("risk_bin")["churn_t"])
    with cols[1]:
        st.dataframe(churn_table)

if step == 'Step 5 – CRM Decision Panel' and 'model' in st.session_state:

    st.header('Step 5: CRM Decision Panel')

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # ========= 基础数据 =========
    risk = model.predict(X_test)

    df_eval = pd.DataFrame({
        "risk": risk,
        "time": y_test["time"],
        "event": y_test["event"]
    })

    # ================================
    # 1️⃣ 客户分层（核心）
    # ================================
    st.subheader("1. Customer Segmentation")

    n_bins = st.slider("Number of segments", 3, 10, 5)

    df_eval["segment"] = pd.qcut(df_eval["risk"], n_bins, labels=[f"S{i+1}" for i in range(n_bins)])

    seg_table = df_eval.groupby("segment").agg(
        customers=("risk", "count"),
        churn_rate=("event", "mean"),
        avg_time=("time", "mean")
    ).reset_index()

    col1, col2 = st.columns([3,2])
    
    col1.bar_chart(seg_table.set_index("segment")["churn_rate"])
    col2.dataframe(seg_table, width='stretch')

    # ================================
    # 2️⃣ 时间窗口流失（业务最重要）
    # ================================
    st.subheader("2. Time-based Churn")

    horizon = st.slider("Time Horizon", 7, 180, 30)

    df_eval["churn_t"] = ((df_eval["time"] <= horizon) & (df_eval["event"] == 1)).astype(int)

    churn_table = df_eval.groupby("segment")["churn_t"].mean().reset_index()

    st.line_chart(churn_table.set_index("segment"))

    # ================================
    # 3️⃣ 干预策略模拟（重点🔥）
    # ================================
    st.subheader("3. Intervention Strategy Simulation")

    top_k = st.slider("Target Top % High Risk Users", 5, 50, 20)

    threshold = np.percentile(df_eval["risk"], 100 - top_k)

    df_eval["target"] = (df_eval["risk"] >= threshold).astype(int)

    col1, col2, col3 = st.columns(3)
    intervention_cost = col1.number_input("Cost per User", value=10.0)
    retention_gain = col2.number_input("Retention Improvement %", value=0.3)

    # ===== ROI计算 =====
    target_users = df_eval[df_eval["target"] == 1]

    base_churn = target_users["churn_t"].mean()
    improved_churn = base_churn * (1 - retention_gain)

    saved_users = len(target_users) * (base_churn - improved_churn)

    total_cost = len(target_users) * intervention_cost
    value_per_user = col3.number_input("Customer Value", value=100.0)

    total_gain = saved_users * value_per_user
    roi = total_gain / total_cost if total_cost > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Target Users", len(target_users))
    col2.metric("Saved Users", f"{saved_users:.1f}")
    col3.metric("ROI", f"{roi:.2f}")

    col1, col2 = st.columns([3,2])
    with col1:
        # ================================
        # 4️⃣ 生存曲线（决策级）
        # ================================
        st.subheader("4. Survival Curve by Segment")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9,5))

        for seg in df_eval["segment"].unique():
            idx = df_eval[df_eval["segment"] == seg].index[:1]
            fn = model.predict_survival_function(X_test.iloc[idx])[0]

            ax.step(fn.x, fn.y, where="post", label=str(seg))

        ax.set_title("Survival Curve by Segment")
        ax.legend()

        st.pyplot(fig, dpi=150)

    with col2:
        # ================================
        # 5️⃣ 自动策略建议（关键🔥）
        # ================================
        st.subheader("5. Recommended Actions")

        def recommend_action(churn_rate):
            if churn_rate > 0.5:
                return "🔥 强干预（优惠券+客服）"
            elif churn_rate > 0.3:
                return "⚠️ 中度干预（触达+提醒）"
            else:
                return "✅ 维持（不打扰）"

        seg_table["action"] = seg_table["churn_rate"].apply(recommend_action)

        st.dataframe(seg_table[["segment", "churn_rate", "action"]], width='stretch')

# =========================================================
# Step 6 - Scoring
# =========================================================
if step == "Step 6 – Scoring" and st.session_state.model is not None:

    st.header("Step 6: Scoring New Dataset")

    # ===================== LOAD ARTIFACT =====================
    artifact = joblib.load("config/survival_preprocess_artifact.pkl")

    id_var = artifact["id_var"]
    time_var = artifact["time_var"]
    event_var = artifact["event_var"]
    features = artifact["features"]

    num_cols = artifact["num_cols"]
    cat_cols = artifact["cat_cols"]
    enc = artifact["ordinal_encoder"]
    num_imputer = artifact["num_imputer"]
    cat_imputer = artifact["cat_imputer"]

    conn = get_conn()
    sources = pd.read_sql("SELECT * FROM data_sources", conn)

    with st.form("score_form"):
        src = st.selectbox("Select a dataset for prediction", sources["name"], index=None)
        run = st.form_submit_button("▶️ Score")

    if run:
        path = sources.loc[sources["name"] == src, "path"].values[0]
        path = path.replace("\\", "/")
        new_df = pd.read_csv(path)

        # ===================== ALIGN FEATURES =====================
        Xnew = new_df.reindex(columns=features)

        # ===================== NUMERIC IMPUTATION =====================
        if len(num_cols) > 0:
            Xnew[num_cols] = num_imputer.transform(Xnew[num_cols])

        # ===================== CATEGORICAL IMPUTATION =====================
        if len(cat_cols) > 0:
            Xnew[cat_cols] = cat_imputer.transform(Xnew[cat_cols].astype(str))

            # encoding（关键：必须用训练encoder）
            Xnew[cat_cols] = enc.transform(Xnew[cat_cols].astype(str))

        # ===================== FINAL CLEAN =====================
        Xnew = Xnew.fillna(0)

        # ===================== SCORING =====================
        # model = st.session_state.model
        model = artifact["model"]
        new_df["risk_score"] = model.predict(Xnew)

        st.write('Scorint Result:')
        st.dataframe(new_df.head(), width='stretch')
        
        st.download_button(
            "⬇️ Download all prediction result",
            data=new_df.to_csv(index=False).encode('utf-8'),
            file_name="survival_scoring_result.csv",
            mime="text/csv",
            key="download_scoring_result"
        )
