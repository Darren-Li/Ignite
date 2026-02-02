import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yaml

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

# import arviz as az
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pymc_marketing.metrics import crps
# from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
# from pymc_marketing.mmm.utils import apply_sklearn_transformer_across_dim
# # from pymc_marketing.prior import Prior
# from pymc_extras.prior import Prior

from services.db_service import get_conn


# =========================
# Utilities
# =========================
def save_yaml(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(obj, f, allow_unicode=True)

def get_k_by_v(dict, value, default=None):
    """返回字典中第一个匹配值的键，未找到返回 default"""
    return next((k for k, v in dict.items() if v == value), default)

def plot_ts_with_target(df, ts, time_freq, target, col):
    # 绘制双坐标轴时序图
    fig = go.Figure()

    # 特征列（左轴）
    if col != target:
        fig.add_trace(go.Scatter(
            x=df[ts],
            y=df[col],
            mode='lines+markers',
            name=col,
            yaxis="y1"
        ))

        # 目标列（右轴）
        fig.add_trace(go.Scatter(
            x=df[ts],
            y=df[target],
            mode='lines+markers',
            name=target,
            yaxis="y2",
            line=dict(dash='dash', color='red')
        ))
    else:
        # 如果当前是目标列，只画自己
        fig.add_trace(go.Scatter(
            x=df[ts],
            y=df[col],
            mode='lines+markers',
            name=col,
            yaxis="y1"
        ))

    # 设置双坐标轴布局
    title = f"{col} 与 {target} 随时间变化" if col != target else f"{target} 随时间变化"
    fig.update_layout(
        title=title,
        xaxis=dict(title="time",
            tickformat="%Y" if time_freq == "年" else
                   "%Y-Q%q" if time_freq == "季度" else
                   "%b-%Y" if time_freq == "月" else
                   "%Y-%m-%d"),
        yaxis=dict(title=col, side='left', showgrid=True),
        yaxis2=dict(title=target, overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50)
    )

    return fig


# =========================
# Streamlit App
# =========================

st.set_page_config(layout='wide')
st.title('Marketing Mix Optimization')

# Sidebar navigation
step = st.sidebar.radio(
    'Workflow',
    ['Step 0 – Data & Variables', 'Step 1 – Exploratory Data Analysis', 'Step 2 – Model Specification',
     'Step 3 – Modeling', 'Step 4 – Media Deep Dive', 'Step 5 – Budget Optimization']
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

    # ========= Load Data =========
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
    col_widths = [1/2, 2, 1, 1*sample_n, 1, 1, 2, 2]
    header_cols = st.columns(col_widths)
    header_texts = ["#", "Field Name", "Type", f"Sample ({sample_n})", "Missing %", "Unique", "Classification", "Operation Type"]

    for col, text in zip(header_cols, header_texts):
        col.markdown(f"**{text}**")

    st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)  # 表头分隔线

    prev_config = st.session_state.get('var_config', {})
    options = ['exclude', 'datetime', 'target', 'control', 'media']
    prev_ops_config = st.session_state.get('ops_config', {})
    ops_options = ['sum', 'mean', 'NA']

    config = {}
    ops_config = {}
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
            if 'datetime' in c.lower() or 'date' in c.lower() or c_type =="datetime64[ns]":
                default_value = 'datetime'
            elif 'target' in c.lower() or '_consumption' in c.lower():
                default_value = 'target'
            elif c.lower().startswith("ctrl_"):
                default_value = 'control'
            else:
                default_value = 'media'
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

        # 先尝试从已有配置取值
        default_ops_value = prev_ops_config.get(c, None)
        # 如果没有，就根据简单规则判断
        if default_ops_value is None:
            if 'datetime' in c.lower() or 'date' in c.lower() or c_type =="datetime64[ns]":
                default_ops_value = 'NA'
            elif 'target' in c.lower():
                default_ops_value = 'sum'
            elif c.lower().startswith("ctrl_"):
                default_ops_value = 'mean'
            else:
                default_ops_value = 'sum'
        # 计算默认索引
        default_ops_index = ops_options.index(default_ops_value) if default_ops_value in ops_options else 0
        
        with cols[7]:
            ops_config[c] = st.selectbox(
                '',
                ops_options,
                index=default_ops_index,
                key=f'ops_{c}_{idx}',
                label_visibility="collapsed"
            )
        # 分隔线
        st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)

    # 保存按钮
    if st.button('💾 Save Variable Configuration'):
        st.session_state.var_config = config
        st.session_state.ops_config = ops_config

        # Path('config').mkdir(exist_ok=True)
        # save_yaml(config, 'config/mmm_config.yaml')
        # save_yaml(ops_config, 'config/mmm_ops_config.yaml')
        # st.success('Variable configuration saved')

# =========================
# Step 1: Exploratory Data Analysis
# =========================

if step == 'Step 1 – Exploratory Data Analysis' and st.session_state.df is not None:
    st.header('Step 1: Exploratory Data Analysis')
    config = st.session_state.var_config
    ops_config = st.session_state.ops_config
    df = st.session_state.df

    # 指定列
    time_col, target_col = get_k_by_v(config, "datetime"), get_k_by_v(config, "target")
    media_vars = [k for k, v in config.items() if v == 'media']
    ctrl_vars = [k for k, v in config.items() if v == 'control']
    feature_cols = media_vars + ctrl_vars
    # feature_cols = [c for c in df.columns if c not in [time_col, target_col]]
    
    df[time_col] = pd.to_datetime(df[time_col])

    # ---------------------------
    # 数据基本信息
    # ---------------------------
    num_rows, num_cols = df.shape
    start_date = df[time_col].min()
    end_date = df[time_col].max()

    st.markdown(f"""
    **数据总览：**
    - 行数: `{num_rows}`, 列数: `{num_cols}`
    - 时间范围: `{start_date}` ~ `{end_date}`
    - 目标变量: `{target_col}`
    """)

    # ---------------------------
    # 时间粒度选择器
    # ---------------------------
    time_freq = st.radio("选择时间聚合粒度",
        [
        # "天",
        "月", "季度", "年"],
        index=0,
        horizontal=True,  # 水平排列
        label_visibility="visible")
    # 重采样映射
    freq_map = {
        # "天": "D",
        "月": "MS",     # 月初
        "季度": "QS",   # 季初
        "年": "YS"      # 年初
    }
    resample_rule = freq_map[time_freq]

    # 按时间重采样
    agg_dict= ops_config
    agg_dict.pop(time_col, None)
    df_resampled = df.set_index(time_col).resample(resample_rule).agg(agg_dict).reset_index()  # 从ops_config获取每个字段聚合函数

    # ---------------------------
    # 变量分析表格
    # ---------------------------
    col_widths_header = [0.5, 2, 1, 1]
    header_cols = st.columns(col_widths_header)
    header_texts = ["#", "Column", "Type", "Operation"]
    for col, text in zip(header_cols, header_texts):
        col.markdown(f"**{text}**")

    st.markdown('<hr style="margin:1px 0; border-top: 2px solid #eee">', unsafe_allow_html=True)

    for idx, col in enumerate([target_col] + feature_cols, 1):
        row1_cols = st.columns(col_widths_header)  # 保持跟表头一样的列数和宽度
        with row1_cols[0]:
            st.write(idx)
        with row1_cols[1]:
            st.write(col)
        with row1_cols[2]:
            st.markdown(f"`{df_resampled[col].dtype}`")
        with row1_cols[3]:
            st.write(ops_config.get(col))

        expanded = True if idx==0 else False
        with st.expander(f"Time Series Chart with `{target_col}`", expanded=expanded):
            fig = plot_ts_with_target(df_resampled, time_col, time_freq, target_col, col)
            st.plotly_chart(fig, use_container_width=True)

        # 每条数据后加分隔线
        st.markdown('<hr style="margin:1px 0; border-top: 1px solid #ddd">', unsafe_allow_html=True)

# =========================
# Step 2: Model Specification
# =========================

if step == 'Step 2 – Model Specification' and st.session_state.df is not None:
    st.header('Step 2. Model Specification')
    df = st.session_state.df
    config = st.session_state.var_config

    media_vars = [k for k, v in config.items() if v == 'media']
    ctrl_vars = [k for k, v in config.items() if v == 'control']

    st.subheader('1. Model Specification')
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        l_max = st.number_input('Max duration', 6, key='p_l_max', help="Maximum duration of carryover effect.")
    with col2:
        yearly_seasonality = st.slider('Yearly seasonality', 2, 6, 4, key=f'p_seasonality',
            help="yearly seasonality")
    with col3:
        time_varying_intercept = st.checkbox(
            "Time varying intercept",     # 必传：复选框旁边的文字说明
            value=False,                  # 可选：默认是否勾选，默认不勾选
            key='p_intercept_vary',       # 可选：绑定session_state的键，用于保留状态
            help="Whether to consider time-varying intercept.")
    col4, col5, col6 = st.columns(3, gap="large")
    with col4:
        target_accept = st.slider('Target accept', 0.7, 1.0, 0.9, key='p_target_accept',
            help="")
    with col5:
        chains = st.number_input('chains', 4, key='p_chains', help="")
    with col6:
        draws = st.number_input('draws', 30000, key='p_draws', help="")

    st.subheader('2. prior_sigma')
    ttl_spend_per_channel = df.tail(24)[media_vars].replace(0, np.nan).mean(axis=0).replace(np.nan,0).copy()
    prior_sigma = ttl_spend_per_channel / ttl_spend_per_channel.sum()
    prior_sigma = prior_sigma.to_numpy()
    st.dataframe(pd.DataFrame({"channel":media_vars,"value":prior_sigma}))

    st.subheader('3. Parameters of Adstock and Saturation')
    col_widths = [1/2, 2, 1, 1, 1, 1]
    header_cols = st.columns(col_widths)
    header_texts = ["#", "Media Name", "Adstock_alpha_alpha", "Adstock_alpha_beta", "Saturation_lam_alpha", "saturation_lam_beta"]

    for col, text in zip(header_cols, header_texts):
        col.markdown(f"**{text}**")

    st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)  # 表头分隔线

    adstock_alpha_a, adstock_alpha_b, saturation_lam_a, saturation_lam_b = [], [], [], []
    for idx, c in enumerate(media_vars, 1):
        cols = st.columns(col_widths)
        # ====== 填充列内容 ======
        with cols[0]:
            st.write(idx)
        with cols[1]:
            st.write(c)
        with cols[2]:
            alpha1 = st.number_input('', 2, key=f'a1_{c}_{idx}')
        with cols[3]:
            alpha2 = st.number_input('', 3, key=f'a2_{c}_{idx}')
        with cols[4]:
            lam1 = st.number_input('', 2, key=f'l1_{c}_{idx}')
        with cols[5]:
            lam2 = st.number_input('', 3, key=f'l2_{c}_{idx}')

        adstock_alpha_a.append(alpha1)
        adstock_alpha_a.append(alpha2)
        saturation_lam_a.append(lam1)
        saturation_lam_a.append(lam2)

        # 分隔线
        st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)

    # 保存按钮
    if st.button('💾 Save Parameters of Adstock and Saturation Configuration'):
        st.session_state.adstock_alpha_a = adstock_alpha_a
        st.session_state.adstock_alpha_b = adstock_alpha_b
        st.session_state.saturation_lam_a = saturation_lam_a
        st.session_state.saturation_lam_b = saturation_lam_b

# =========================
# Step 3: Modeling
# =========================

if step == 'Step 3 – Modeling' and st.session_state.df is not None:
    st.header('Step 3: Modeling')

    df = st.session_state.df
    config = st.session_state.var_config

    media_vars = [k for k, v in config.items() if v == 'media']
    ctrl_vars = [k for k, v in config.items() if v == 'control']

    ctrl_vars_std = []
    scalers = dict()
    scaled_data = pd.DataFrame()
    for col in ctrl_vars:
        scaler = MaxAbsScaler()
        #scaler = StandardScaler()
        scalers[col] = scaler.fit(df[[col]])
        scaled_data[col] = scalers[col].transform(df[[col]]).flatten()

    for i, col in enumerate(ctrl_vars):
        ctrl_vars_std.append(f'{col}_std')
        df[f'{col}_std'] = scaled_data.iloc[:, i]

    model_config = {
        'intercept': Prior("Normal", mu=0, sigma=1),
        'likelihood': Prior("Normal", sigma=Prior("HalfNormal", sigma=1)),
        'gamma_control': Prior("Normal", mu=0, sigma=.1, dims="control"),
        'gamma_fourier': Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        'adstock_alpha': Prior("Beta", alpha=adstock_alpha_alpha, beta=adstock_alpha_beta, dims="channel"),
        'saturation_lam': Prior("Gamma", alpha=saturation_lam_alpha, beta=saturation_lam_beta, dims="channel"),
        'saturation_beta': Prior("HalfNormal", sigma=prior_sigma, dims="channel")
    }

    # ----------------- 训练按钮 -----------------
    run_training = st.button("🚀 Train Model")
    
    if run_training:
        sampler_config = {"progressbar": False}
        mmm = MMM(
            model_config=model_config,
            sampler_config=sampler_config,
            date_column=date_column,
            adstock=GeometricAdstock(l_max=12),
            saturation=LogisticSaturation(),
            channel_columns=media_vars,
            control_columns=ctrl_vars_std,
            yearly_seasonality=6,
            time_varying_intercept=True
        )
        # ---------------- Fit ----------------
        mmm.fit(
            X=df[media_vars+ctrl_vars_std],
            y=df[target],
            target_accept=0.9,
            chains=4,
            draws=3_000,
            # nuts_sampler="numpyro",
            random_seed=rng
            )

        # ---------------- Save ----------------
        st.session_state.model = mmm
        st.success(f"Model trained: {model_name}")

# =========================
# Step 4: Media Deep Dive
# =========================
if step == 'Step 4 – Media Deep Dive' and 'model' in st.session_state:
    st.header('Step 4: Media Deep Dive')

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
            st.markdown("**Lift Chart**")
            st.line_chart(lift_df.set_index('decile')[['lift']])
        with col2:
            st.markdown("**Cumulative Lift**")
            st.line_chart(lift_df.set_index('decile')[['cum_lift']])
        
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
        gain_df = build_gain_table(
            y_true=y_test.values,
            y_score=pred,
            n_bins=n_bins
        )

        st.markdown("**Cumulative Gain Table**")
        st.dataframe(gain_df, hide_index=True, use_container_width=True)

        st.markdown("**Cumulative Gain Chart**")
        st.line_chart(gain_df.set_index('decile')[['cum_gain_pct']])

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
                positives=('y_true', 'sum')
            )
            .reset_index()
        )

        table['positive_rate'] = table['positives'] / table['total']
        overall_rate = df['y_true'].mean()
        table['lift'] = table['positive_rate'] / overall_rate

        table['cum_positives'] = table['positives'].cumsum()
        table['cum_total'] = table['total'].cumsum()
        table['cum_positive_rate'] = table['cum_positives'] / table['cum_total']
        table['cum_lift'] = table['cum_positive_rate'] / overall_rate

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
                total_value=('y_true', 'sum'),
                count=('y_true', 'count')
            )
            .reset_index()
        )

        table['cum_value'] = table['total_value'].cumsum()
        total_value = table['total_value'].sum()
        table['cum_gain_pct'] = table['cum_value'] / total_value

        return table

    evaluate_model(
        model=st.session_state.model,
        X_test=st.session_state.X_test,
        y_test=st.session_state.y_test,
        task_type=st.session_state.task_type
    )

# =========================
# Step 5: Budget Optimization
# =========================
if step == 'Step 5 – Budget Optimization' and 'model' in st.session_state:
    st.header('Step 5: Budget Optimization')

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

        with open('config/preprocess_pipeline.yaml', 'r', encoding='utf-8') as f:
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
