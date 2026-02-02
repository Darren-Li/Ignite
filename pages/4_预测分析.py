import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
from pathlib import Path
import yaml

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

from services.db_service import get_conn


# =========================
# Utilities
# =========================
def save_yaml(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(obj, f, allow_unicode=True)


# =========================
# Streamlit App
# =========================

st.set_page_config(layout='wide')
st.title('Predictive Analytics')

# Sidebar navigation
step = st.sidebar.radio(
    'Workflow',
    ['Step 0 – Data & Variables', 'Step 1 – Preprocessing', 'Step 2 – Variable Selection',
     'Step 3 – Modeling', 'Step 4 – Evaluation', 'Step 5 – Scoring']
)

# Shared state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'var_config' not in st.session_state:
    st.session_state.var_config = {}
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

# =========================
# Step 0: Data & Variable Setup
# =========================

if step == 'Step 0 – Data & Variables':
    st.header('Step 0: Data Import & Variable Classification')
    st.subheader('1. Load Data')

    conn = get_conn()
    sources = pd.read_sql("SELECT * FROM data_sources", conn)

    # ========= 显式选择 + 提交 =========
    with st.form("data_select_form"):
        src = st.selectbox(
            "Select a dataset for modeling",
            sources["name"],
            index=None,
            placeholder="Select a dataset..."
        )
        submitted = st.form_submit_button("▶️ Load Data and Analyse")

    # ========= 只在提交时读数据 =========
    if submitted:
        path = sources.loc[sources["name"] == src, "path"].values[0]

        try:
            df = pd.read_csv(path)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

        st.session_state.df = df
        st.session_state.selected_source = src

    # ========= 关键防线（不是可选） =========
    if (
        'df' not in st.session_state
        or st.session_state.df is None
        or not isinstance(st.session_state.df, pd.DataFrame)
    ):
        st.info("Please select a dataset and click **Load Data** to continue.")
        st.stop()

    # ========= 从这里开始，df 100% 安全 =========
    df = st.session_state.df
    st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ========= Profiling & Classification =========
    st.subheader('2. Variable Profiling & Classification')

    sample_n = 3
    col_widths = [1/2, 2, 1, 1*sample_n, 1, 1, 2]
    header_cols = st.columns(col_widths)
    header_texts = ["#", "Field Name", "Type", f"Sample ({sample_n})", "Missing %", "Unique", "Classification"]

    for col, text in zip(header_cols, header_texts):
        col.markdown(f"**{text}**")

    st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)  # 表头分隔线

    prev_config = st.session_state.get('var_config', {})
    options = ['exclude', 'target', 'id', 'binary', 'nominal', 'ordinal', 'continuous']

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
            if '_id' in c.lower():
                default_value = 'id'
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
                '',
                options,
                index=default_index,
                key=f'var_{c}_{idx}',
                label_visibility="collapsed"
            )

        # 分隔线
        st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)

    # 保存按钮
    if st.button('💾 Save Variable Configuration'):
        st.session_state.var_config = config
        Path('config').mkdir(exist_ok=True)
        save_yaml(config, 'config/prediction_variables.yaml')
        st.success('Variable configuration saved')

# =========================
# Step 1: Preprocessing
# =========================

if step == 'Step 1 – Preprocessing' and st.session_state.df is not None:
    st.header('Step 1: Preprocessing')
    config = st.session_state.var_config

    st.subheader('1. Parameter Configuration')
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        misspct = st.slider('Max missing rate', 0.1, 0.9, 0.75, 
            help="Maximum missing percent for a variable. This cannot be greater than .5 if using Hotdeck for imputation;")
    with col2:
        maxpct = st.slider('Max concentration ratio', 0.5, 0.95, 0.9, 
            help="Maximum percent in a single bucket for ordinal and nominal variables;")
    with col3:
        maxcat = st.number_input('Max category', value=25, 
            help="Maximum number of values for ordinal variables. Any variable over this will be re-classified to continuous;")
    with col4:
        maxcorr = st.slider('Max correlation', 0.3, 0.95, 0.7, 
            help="maximum correlation between variables")

    col5, col6 = st.columns(2, gap="large")
    with col5:
        impute_methodO = st.selectbox('Imputation Method for Ordinal variables', ['median', 'knn'])
    with col6:
        impute_method = st.selectbox('Imputation Method for Continues variables', ['mean', 'median', 'knn'])
    
    cont_vars = [k for k, v in config.items() if v == 'continuous']
    ord_vars = [k for k, v in config.items() if v == 'ordinal']
    nom_vars = [k for k, v in config.items() if v == 'nominal']
    bin_vars = [k for k, v in config.items() if v == 'binary']
    id_var = [k for k, v in config.items() if v == 'id']
    target_var = [k for k, v in config.items() if v == 'target']

    feat_vars = cont_vars + ord_vars + nom_vars + bin_vars
    df = st.session_state.df[id_var+target_var+feat_vars].copy()

    st.subheader('2. Dropped Variables')
    # high missing rate
    drop_vars_mis = [c for c in feat_vars if df[c].isna().mean() > misspct]
    # concentration ratio
    drop_vars_max = [c for c in ord_vars+nom_vars if df[c].value_counts(normalize=True).iloc[0] > maxpct]

    # max category
    ord_vars_cat = [c for c in ord_vars if len(df[c].dropna().unique()) > maxcat]
    # remove vars from ord_vars
    ord_vars = [x for x in ord_vars if x not in ord_vars_cat]
    # add vars into cont_vars
    cont_vars.extend(ord_vars_cat)

    drop_vars = drop_vars_mis + drop_vars_max
    df = df.drop(columns=drop_vars)

    st.write('Dropped variables due to missing and concentration', drop_vars)

    # update features
    existing_cols = set(df.columns)
    cont_vars = [c for c in cont_vars if c in existing_cols]
    ord_vars  = [c for c in ord_vars  if c in existing_cols]
    nom_vars  = [c for c in nom_vars  if c in existing_cols]
    bin_vars  = [c for c in bin_vars  if c in existing_cols]

    feat_vars = cont_vars + ord_vars + nom_vars + bin_vars

    st.subheader('3. Preprocessing')
    # ----------------- 训练按钮 -----------------
    run_preprocessing = st.button("🚀 Preprocessing")
    
    if run_preprocessing:
        # Nominal recode
        nominal_params = {}
        if nom_vars:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[nom_vars] = enc.fit_transform(df[nom_vars].astype(str))
            for i, col in enumerate(nom_vars):
                nominal_params[col] = {'categories': enc.categories_[i].tolist()}

        ordinal_params = {}
        if ord_vars:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[ord_vars] = enc.fit_transform(df[ord_vars].astype(str))
            for i, col in enumerate(ord_vars):
                ordinal_params[col] = {'categories': enc.categories_[i].tolist()}

        # Binary recode
        binary_params = {}
        for col in bin_vars:
            true_vals = ['1','y','yes','true']
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() in true_vals else 0)
            binary_params[col] = {'true_values': true_vals}

        # Continues recode and Imputation
        # 1. capping floor
        # 2. Imputation
        continuous_params = {}
        for col in cont_vars:
            # 填充
            if impute_method == 'mean':
                impute_value = df[col].mean()
                df[col].fillna(impute_value, inplace=True)
                fill_value = impute_value
            elif impute_method == 'median':
                impute_value = df[col].median()
                df[col].fillna(impute_value, inplace=True)
                fill_value = impute_value
            # KNNImputer 也可以单独拟合，但这里简单示例用 median

            # 标准化
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std

            # 保存参数
            continuous_params[col] = {
            'fill_value': float(fill_value), 'mean': float(mean), 'std': float(std)
            }

        # 保存到 YAML
        preprocess_pipeline = {
            'continuous': continuous_params,
            'nominal': nominal_params,
            'ordinal': ordinal_params,
            'binary': binary_params
        }

        with open('config/prediction_preprocess_pipeline.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(preprocess_pipeline, f, allow_unicode=True)

        st.success("Preprocessing completed and pipeline saved")

        st.session_state.processed_df = df

        st.write('The preprocessed dataset')
        st.dataframe(df.head(), use_container_width=True)

# =========================
# Step 2: Variable Selection
# =========================

if step == 'Step 2 – Variable Selection' and st.session_state.processed_df is not None:
    st.header('Step 2: Variable Selection')
    df = st.session_state.processed_df
    config = st.session_state.var_config

    id_var = [k for k, v in config.items() if v == 'id']
    target_var = [k for k, v in config.items() if v == 'target']

    howmany = st.slider('Number of variables to keep',
        math.ceil(df.shape[1] * 0.5), min(50, df.shape[1]), 
        math.ceil(df.shape[1] * 0.8)
        )

    variances = df.drop(columns=id_var + target_var, errors='ignore').var().sort_values(ascending=False)
    selected = variances.head(howmany).index.tolist()

    st.write('Selected variables')
    st.write(selected)

    st.session_state.selected_vars = selected
    df = df[id_var + target_var + selected].copy()
    st.session_state.processed_df = df

    st.write('The final dataset used for modeling')
    st.dataframe(df.head(), use_container_width=True)

# =========================
# Step 3: Model selection and tuning
# =========================

if step == 'Step 3 – Modeling' and st.session_state.processed_df is not None:
    st.header('Step 3: Model Selection & Training')

    df = st.session_state.processed_df
    config = st.session_state.var_config

    target_var = [k for k, v in config.items() if v == 'target']
    id_var = [k for k, v in config.items() if v == 'id']

    if len(target_var) != 1:
        st.error("Exactly one target variable must be specified.")
        st.stop()

    target = target_var[0]
    X = df.drop(columns=id_var + target_var, errors='ignore')
    y = df[target]

    col1, col2 = st.columns(2)
    with col1:
        task_type = st.radio(
            "Prediction Type",
            ['Binary Classification', 'Regression'],
            key='task_type_ui'
        )
    with col2:
        if task_type == 'Binary Classification':
            model_name = st.selectbox(
                "Model",
                ['Logistic Regression', 'Random Forest', 'XGBoost Classifier'],
                key='model_name_ui'
            )
        else:
            model_name = st.selectbox(
                "Model",
                ['Linear Regression', 'XGBoost Regression', 'Random Forest Regressor'],
                key='model_name_ui'
            )

    test_size = st.slider(
        'Test size', 0.1, 0.5, 0.3,
        key='test_size_ui'
    )

    # ----------------- 训练按钮 -----------------
    run_training = st.button("🚀 Train Model")
    
    if run_training:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42
        )

        # ---------------- Train ----------------
        if model_name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
        elif model_name == 'Random Forest' and task_type == 'Binary Classification':
            model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        elif model_name == 'XGBoost Classifier':
            model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        elif model_name == 'Linear Regression':
            model = LinearRegression()
        elif model_name == 'XGBoost Regression':
            model = XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42
            )
        elif model_name == 'Random Forest Regressor':
            model = RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=42
            )

        # ---------------- Fit ----------------
        model.fit(X_train, y_train)

        # ---------------- Save ----------------
        st.session_state.model = model
        st.session_state.task_type = task_type
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.features = X.columns.tolist()

        st.success(f"Model trained: {model_name}")

# =========================
# Step 4: Evaluation
# =========================
if step == 'Step 4 – Evaluation' and 'model' in st.session_state:
    st.header('Step 4: Model Evaluation')

    def evaluate_model(model, X_test, y_test, task_type):
        if task_type == 'Binary Classification':
            evaluate_classification(model, X_test, y_test)
        else:
            evaluate_regression(model, X_test, y_test)

    def evaluate_classification(model, X_test, y_test, n_bins=10):
        st.subheader("Binary Classification Evaluation")

        prob = model.predict_proba(X_test)[:, 1]

        # ===== 基础指标 =====
        auc = roc_auc_score(y_test, prob)
        fpr, tpr, _ = roc_curve(y_test, prob)
        ks = max(tpr - fpr)

        c1, c2 = st.columns(2)
        c1.metric("AUC", f"{auc:.3f}")
        c2.metric("KS", f"{ks:.3f}")

        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        st.line_chart(roc_df)

        # ===== Lift / Cumulative Lift =====
        lift_df = build_lift_table(
            y_true=y_test.values,
            y_score=prob,
            n_bins=n_bins
        )

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("**Lift Chart (Response Rate Index)**")
            st.line_chart(lift_df.set_index('decile')[['lift', 'cum_lift']])
        with col2:
            st.markdown("**Response Rate**")
            st.line_chart(lift_df.set_index('decile')[['positive_rate', 'cum_positive_rate']])
        
        col3, col4 = st.columns(2, gap="medium")
        with col3:
            st.markdown("**Response Ratio**")
            st.line_chart(lift_df.set_index('decile')[['positives_ratio', 'cum_positives_ratio']])
        with col4:
            st.markdown("**Response Rate - Predicted vs. Actual**")
            st.line_chart(lift_df.set_index('decile')[['pred_positive_rate', 'positive_rate']])
            
        st.markdown("**Lift Table**")
        st.dataframe(lift_df, hide_index=True, use_container_width=True)
        
    def evaluate_regression(model, X_test, y_test, n_bins=10):
        st.subheader("Regression Evaluation")

        pred = model.predict(X_test)

        # ===== 误差指标 =====
        rmse = mean_squared_error(y_test, pred, squared=False)
        r2 = r2_score(y_test, pred)

        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{rmse:.3f}")
        c2.metric("R²", f"{r2:.3f}")

        # ===== Gain / Cumulative Gain =====
        lift_df = build_gain_table(
            y_true=y_test.values,
            y_score=pred,
            n_bins=n_bins
        )

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("**Lift Chart (Revenue Index)**")
            st.line_chart(lift_df.set_index('decile')[['lift', 'cum_lift']])
        with col2:
            st.markdown("**Average Revenue**")
            st.line_chart(lift_df.set_index('decile')[['rev_mean', 'cum_rev_mean']])
        
        col3, col4 = st.columns(2, gap="medium")
        with col3:
            st.markdown("**Average Revenue - Predicted vs. Actual**")
            st.line_chart(lift_df.set_index('decile')[['pred_rev_mean', 'rev_mean']])

        st.markdown("**Cumulative Lift Table**")
        st.dataframe(lift_df, hide_index=True, use_container_width=True)

    def build_lift_table(y_true, y_score, n_bins=10):
        df = pd.DataFrame({
            'y_true': y_true,
            'y_score': y_score
        }).sort_values('y_score', ascending=False).reset_index(drop=True)

        df['decile'] = pd.qcut(
            df.index + 1,
            n_bins,
            labels=[f'D{i+1}' for i in range(n_bins)]
        )

        table = (
            df.groupby('decile')
            .agg(
                total=('y_true', 'count'),
                positives=('y_true', 'sum'),
                pred_positive_rate=('y_score', 'mean')
            )
            .reset_index()
        )

        table['positive_rate'] = table['positives'] / table['total']
        overall_rate = df['y_true'].mean()
        table['lift'] = table['positive_rate'] / overall_rate

        table['cum_positives'] = table['positives'].cumsum() 
        table['cum_positive_rate'] = table['cum_positives'] / table['total'].cumsum()
        table['cum_lift'] = table['cum_positive_rate'] / overall_rate

        overall_positives = df['y_true'].sum()
        table['positives_ratio'] = table['positives'] / overall_positives
        table['cum_positives_ratio'] = table['positives_ratio'].cumsum()

        return table

    def build_gain_table(y_true, y_score, n_bins=10):
        df = pd.DataFrame({
            'y_true': y_true,
            'y_score': y_score
        }).sort_values('y_score', ascending=False).reset_index(drop=True)

        df['decile'] = pd.qcut(
            df.index + 1,
            n_bins,
            labels=[f'D{i+1}' for i in range(n_bins)]
        )

        table = (
            df.groupby('decile')
            .agg(
                total = ('y_true', 'count'),
                rev_sum=('y_true', 'sum'),
                rev_mean=('y_true', 'mean'),
                pred_rev_mean=('y_score', 'mean')
            )
            .reset_index()
        )

        overall_rev_mean = df["y_true"].mean()
        table['lift'] = table['rev_mean'] / overall_rev_mean

        table['cum_rev_sum'] = table['rev_sum'].cumsum()
        table['cum_rev_mean'] = table['cum_rev_sum'] / table['total'].cumsum()
        table['cum_lift'] = table['cum_rev_mean'] / overall_rev_mean

        return table

    evaluate_model(
        model=st.session_state.model,
        X_test=st.session_state.X_test,
        y_test=st.session_state.y_test,
        task_type=st.session_state.task_type
    )

# =========================
# Step 5: Scoring
# =========================
if step == 'Step 5 – Scoring' and 'model' in st.session_state:
    st.header('Step 5: Scoring New Dataset')

    conn = get_conn()
    sources = pd.read_sql("SELECT * FROM data_sources", conn)

    with st.form("scoring_form"):
        src = st.selectbox(
            "Select a dataset for prediction",
            sources["name"],
            index=None
        )
        submitted = st.form_submit_button("▶️ Run")

    if submitted and src is not None:
        path = sources.loc[sources["name"] == src, "path"].values[0]

        config = st.session_state.var_config
        id_var = [k for k, v in config.items() if v == 'id']
        features = st.session_state.features

        new_df = pd.read_csv(path, usecols=id_var + features)
        st.dataframe(new_df.head())

        with open('config/prediction_preprocess_pipeline.yaml', 'r', encoding='utf-8') as f:
            pipeline = yaml.safe_load(f)

        # ---------- preprocessing ----------
        for col, params in pipeline.get('nominal', {}).items():
            if col in new_df:
                enc = OrdinalEncoder(
                    categories=[params['categories']],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                new_df[[col]] = enc.fit_transform(new_df[[col]].astype(str))

        for col, params in pipeline.get('binary', {}).items():
            if col in new_df:
                new_df[col] = new_df[col].apply(
                    lambda x: 1 if str(x).lower() in params['true_values'] else 0
                )

        for col, params in pipeline.get('continuous', {}).items():
            if col in new_df:
                new_df[col].fillna(params['fill_value'], inplace=True)
                new_df[col] = (new_df[col] - params['mean']) / params['std']

        Xnew = new_df[features]
        model = st.session_state.model

        if st.session_state.task_type == 'Binary Classification':
            new_df['prediction'] = model.predict_proba(Xnew)[:, 1]
        else:
            new_df['prediction'] = model.predict(Xnew)

        st.dataframe(new_df.head(), use_container_width=True)

        st.download_button(
            "💾 Download prediction result",
            data=new_df.to_csv(index=False).encode('utf-8'),
            file_name="prediction_result.csv",
            mime="text/csv"
        )
