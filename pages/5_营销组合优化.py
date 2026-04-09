# import warnings
import streamlit as st
import os
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import yaml
import uuid

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pymc as pm
import seaborn as sns
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.metrics import crps
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)

seed: int = sum(map(ord, "mmm"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
# warnings.filterwarnings("ignore", category=FutureWarning)

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
                'select the processing method for the feature or the type of feature',
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
                'select the function for aggregating the feature',
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
    行数: `{num_rows}`, 列数: `{num_cols}`;      时间范围: `{start_date}` ~ `{end_date}`;      目标变量: `{target_col}`
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
            st.plotly_chart(fig, width="stretch")

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
    cols = st.columns(3, gap="large")
    with cols[0]:
        l_max = st.number_input('Max duration', 6, key='p_l_max', help="Maximum duration of carryover effect.")
    with cols[1]:
        yearly_seasonality = st.slider('Yearly seasonality', 2, 6, 4, key=f'p_seasonality',
            help="yearly seasonality")
    with cols[2]:
        time_varying_intercept = st.checkbox(
            "Time varying intercept",     # 必传：复选框旁边的文字说明
            value=False,                  # 可选：默认是否勾选，默认不勾选
            key='p_intercept_vary',       # 可选：绑定session_state的键，用于保留状态
            help="Whether to consider time-varying intercept.")

    st.session_state.model_config_parameters = [l_max, yearly_seasonality, time_varying_intercept]

    cols = st.columns(4, gap="large")
    with cols[0]:
        target_accept = st.slider('Target accept', 0.7, 1.0, 0.9, key='p_target_accept', help="")
    with cols[1]:
        chains = st.number_input('chains', 2, 6, 4, key='p_chains', help="独立的 MCMC 链数量。常用 4~6 条链，太多会消耗计算资源。")
    with cols[2]:
        draws = st.number_input('draws', 500, key='p_draws', help="每条链从 后验分布中采样的有效样本数量。")
    with cols[3]:
        tune = st.number_input('tune', 1000, key='p_tune', help="热身 / 调整期（tuning / burn-in）的步数。热身步数通常 ≥ draws。")

    st.session_state.model_fit_parameters = [target_accept, chains, draws, tune]

    st.subheader('2. prior_sigma')
    ttl_spend_per_channel = df.tail(24)[media_vars].replace(0, np.nan).mean(axis=0).replace(np.nan,0).copy()
    prior_sigma = ttl_spend_per_channel / ttl_spend_per_channel.sum()
    prior_sigma = prior_sigma.to_numpy()
    st.dataframe(pd.DataFrame({"channel":media_vars,"value":prior_sigma}), width='content')

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
            alpha1 = st.number_input('set the parameter values', 2, key=f'a1_{c}_{idx}', label_visibility="hidden")
        with cols[3]:
            alpha2 = st.number_input('set the parameter values', 3, key=f'a2_{c}_{idx}', label_visibility="hidden")
        with cols[4]:
            lam1 = st.number_input('set the parameter values', 2, key=f'l1_{c}_{idx}', label_visibility="hidden")
        with cols[5]:
            lam2 = st.number_input('set the parameter values', 3, key=f'l2_{c}_{idx}', label_visibility="hidden")

        adstock_alpha_a.append(alpha1)
        adstock_alpha_b.append(alpha2)
        saturation_lam_a.append(lam1)
        saturation_lam_b.append(lam2)

        # 分隔线
        st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)

    # 保存按钮
    if st.button('💾 Save Parameters of Adstock and Saturation Configuration'):
        st.session_state.prior_sigma = prior_sigma
        st.session_state.adstock_alpha_a = adstock_alpha_a
        st.session_state.adstock_alpha_b = adstock_alpha_b
        st.session_state.saturation_lam_a = saturation_lam_a
        st.session_state.saturation_lam_b = saturation_lam_b


# =========================
# Step 3: Modeling
# =========================

if step == 'Step 3 – Modeling' and st.session_state.df is not None:
    st.header('Step 3: Modeling')
    st.subheader('1. Model training or Selection')
    
    # 确保目录存在
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    # --- 1. 获取现有模型列表 ---
    # 扫描目录下所有以 .nc 结尾的文件
    available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".nc")]
    # 如果有模型，允许选择；如果没有，默认为训练
    if available_models:
        mode = st.radio(
            "选择操作模式：",
            options=["加载现有模型", "训练新模型"],
            index=0  # 默认选中第一个
        )
    else:
        st.info("未检测到现有模型，请进行训练。")
        mode = "训练新模型"

    if mode == "加载现有模型":
        st.write("🧠 加载现有模型")
        
        # 下拉菜单选择具体模型文件
        selected_model_file = st.selectbox(
            "选择要加载的模型文件：",
            options=available_models,
            format_func=lambda x: x  # 显示文件名
        )
        
        model_path = os.path.join(MODEL_DIR, selected_model_file)
        
        # 显示加载按钮
        if st.button(f"加载模型：{selected_model_file}"):
            try:
                with st.spinner(f"正在从 {model_path} 加载模型..."):
                    # 核心加载代码
                    # 注意：这里假设你之前保存的是 mmm 对象
                    # 如果你的 MMM 类定义在不同模块，请确保环境一致
                    loaded_mmm = MMM.load(model_path)
                    
                st.success("✅ 模型加载成功！")
                
                # 将加载的模型存入 session_state，供后续页面使用
                st.session_state['model'] = loaded_mmm
                st.session_state['model_source'] = "loaded"
                
                # 简单展示模型信息（可选）
                st.write(f"**模型来源:** {model_path}")
                # 如果有先验图或数据概览，可以在这里展示
                # loaded_mmm.plot_components_contributions()
                
            except Exception as e:
                st.error(f"❌ 加载失败：{e}")

    elif mode == "训练新模型":
        st.write("🔢 训练新模型")

        df = st.session_state.df
        config = st.session_state.var_config

        time_col, target_col = get_k_by_v(config, "datetime"), get_k_by_v(config, "target")
        media_vars = [k for k, v in config.items() if v == 'media']
        ctrl_vars = [k for k, v in config.items() if v == 'control']

        # 对控制变量进行标准化处理
        # ctrl_vars_std = []
        # scalers = dict()
        # scaled_data = pd.DataFrame()
        # for col in ctrl_vars:
        #     scaler = MaxAbsScaler()
        #     #scaler = StandardScaler()
        #     scalers[col] = scaler.fit(df[[col]])
        #     scaled_data[col] = scalers[col].transform(df[[col]]).flatten()

        # for i, col in enumerate(ctrl_vars):
        #     ctrl_vars_std.append(f'{col}_std')
        #     df[f'{col}_std'] = scaled_data.iloc[:, i]

        model_config = {
            'intercept': Prior("Normal", mu=0, sigma=1),
            'likelihood': Prior("Normal", sigma=Prior("HalfNormal", sigma=1)),
            'gamma_control': Prior("Normal", mu=0, sigma=.1, dims="control"),
            'gamma_fourier': Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
            'adstock_alpha': Prior("Beta", alpha=st.session_state.adstock_alpha_a, beta=st.session_state.adstock_alpha_b, dims="channel"),
            'saturation_lam': Prior("Gamma", alpha=st.session_state.saturation_lam_a, beta=st.session_state.saturation_lam_b, dims="channel"),
            'saturation_beta': Prior("HalfNormal", sigma=st.session_state.prior_sigma, dims="channel")
        }

        # ----------------- 训练按钮 -----------------
        run_training = st.button("🚀 Train Model")
        
        if run_training:
            progress_placeholder = st.empty()
            message_placeholder = st.empty()

            message_placeholder.info("⏳ 正在准备数据与划分训练集...")
            train_test_split_date = pd.to_datetime("2025-01-01")

            train_mask = df[time_col] < train_test_split_date
            test_mask = df[time_col] >= train_test_split_date

            train_df = df[train_mask]
            test_df = df[test_mask]

            st.session_state.train_df = train_df
            st.session_state.test_df = test_df

            cols = st.columns(2)
            with cols[0]:
                st.write("Sales - Train Test Split")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.lineplot(data=train_df, x=time_col, y=target_col, color="C0", label="Train", ax=ax)
                sns.lineplot(data=test_df, x=time_col, y=target_col, color="C1", label="Test", ax=ax)
                ax.set(xlabel="date")
                ax.set_title("Sales - Train Test Split", fontsize=16, fontweight="bold")
                st.pyplot(fig, dpi=150)

            with cols[1]:
                st.write(f"Training set: {train_df.shape[0]} observations")
                st.write(f"Test set: {test_df.shape[0]} observations")

            X_train = train_df.drop(columns=target_col)
            X_test = test_df.drop(columns=target_col)

            y_train = train_df[target_col]
            y_test = test_df[target_col]

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            message_placeholder.info("🏗️ 正在构建模型结构...")
            sampler_config = {"progressbar": True}
            model_config_parameters = st.session_state.model_config_parameters

            mmm = MMM(
                model_config=model_config,
                sampler_config=sampler_config,
                target_column=target_col,
                date_column=time_col,
                adstock=GeometricAdstock(l_max=model_config_parameters[0]),
                saturation=LogisticSaturation(),
                channel_columns=media_vars,
                control_columns=ctrl_vars,
                yearly_seasonality=model_config_parameters[1],
                time_varying_intercept=model_config_parameters[2]
            )

            message_placeholder.warning("🚀 模型正在训练中 (MCMC Sampling)... 请勿关闭页面")
            mmm.build_model(X_train, y_train)

            # Add the contribution variables to the model
            # to track them in the model and trace.
            mmm.add_original_scale_contribution_variable(
                var=[
                    "channel_contribution",
                    "control_contribution",
                    "intercept_contribution",
                    "yearly_seasonality_contribution",
                    "y",
                ]
            )

            # pm.model_to_graphviz(mmm.model)
            
            # ---------------- Fit ----------------
            model_fit_parameters = st.session_state.model_fit_parameters
            mmm.fit(
                X=X_train,
                y=y_train,
                target_accept=model_fit_parameters[0],
                chains=model_fit_parameters[1],
                draws=model_fit_parameters[2],
                tune=model_fit_parameters[3],
                nuts_sampler="nutpie", # nutpie, pymc, numpyro
                random_seed=rng,
            )

            # mmm.sample_posterior_predictive(X=X_train, random_seed=rng)

            # ---------------- Save ----------------
            message_placeholder.info("💾 正在保存模型...")
            st.session_state['model'] = mmm
            st.session_state['model_source'] = "trained"

            model_path = os.path.join(MODEL_DIR, f"model_MMM_{uuid.uuid4().hex[:6]}.nc")
            mmm.save(model_path)
            st.success(f"Model trained and saved successfully. \n Model Path: {model_path}")


# =========================
# Step 4: Media Deep Dive
# =========================
if step == 'Step 4 – Media Deep Dive' and 'model' in st.session_state:
    st.header('Step 4: Media Deep Dive')

    def plot_channel_contributions(mmm, df, date, target):
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.lineplot(
            data=df, x=date, y=target, color="black", label="Sales", ax=ax
        )

        for i, hdi_prob in enumerate([0.94, 0.5]):
            az.plot_hdi(
                x=mmm.model.coords["date"],
                y=mmm.idata["posterior"]["channel_contribution_original_scale"].sum(
                    dim="channel"
                ),
                color="C0",
                smooth=False,
                hdi_prob=hdi_prob,
                fill_kwargs={
                    "alpha": 0.3 + i * 0.1,
                    "label": f"{hdi_prob:.0%} HDI (Channels Contribution)",
                },
                ax=ax,
            )

            az.plot_hdi(
                x=mmm.model.coords["date"],
                y=mmm.idata["posterior"]["control_contribution_original_scale"].sum(
                    dim="control"
                ),
                color="C1",
                smooth=False,
                hdi_prob=hdi_prob,
                fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob:.0%} HDI Control"},
                ax=ax,
            )

            az.plot_hdi(
                x=mmm.model.coords["date"],
                y=mmm.idata["posterior"]["yearly_seasonality_contribution_original_scale"],
                color="C2",
                smooth=False,
                hdi_prob=hdi_prob,
                fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob:.0%} HDI Fourier"},
                ax=ax,
            )

            az.plot_hdi(
                x=mmm.model.coords["date"],
                y=mmm.idata["posterior"]["intercept_contribution_original_scale"]
                .expand_dims({"date": mmm.model.coords["date"]})
                .transpose(..., "date"),
                color="C3",
                smooth=False,
                hdi_prob=hdi_prob,
                fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob:.0%} HDI Intercept"},
                ax=ax,
            )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
        )

        fig.suptitle(
            "Posterior Predictive - Channel Contributions",
            fontsize=16,
            fontweight="bold",
            y=1.03,
        )

        return fig, ax

    def plot_channel_contribution_share(channel_contribution_share, channel_columns, spend_shares):
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, channel in enumerate(channel_columns):
            # Contribution share mean and hdi
            share_mean = channel_contribution_share.sel(channel=channel).mean().to_numpy()
            share_hdi = az.hdi(channel_contribution_share.sel(channel=channel))[
                "channel_contribution_original_scale"
            ].to_numpy()

            # ROAS mean and hdi
            roas_mean = roas.sel(channel=channel).mean().to_numpy()
            roas_hdi = az.hdi(roas.sel(channel=channel))["roas"].to_numpy().flatten()

            # Plot the contribution share hdi
            ax.vlines(share_mean, roas_hdi[0], roas_hdi[1], color=f"C{i}", alpha=0.8)

            # Plot the ROAS hdi
            ax.hlines(roas_mean, share_hdi[0], share_hdi[1], color=f"C{i}", alpha=0.8)

            # Plot the means
            ax.scatter(
                share_mean,
                roas_mean,
                # Size of the scatter points is proportional to the spend share
                s=5 * (spend_shares[i] * 100),
                color=f"C{i}",
                edgecolor="black",
                label=channel,
            )
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", title="Channel", title_fontsize=14
        )
        ax.set(
            title="Channel Contribution Share vs ROAS",
            xlabel="Contribution Share",
            ylabel="ROAS",
        )

        return fig, ax

    def plot_predictions(mmm, X_test, y_test, date_column):
        y_pred_test = mmm.sample_posterior_predictive(
            X_test,
            include_last_observations=True,
            var_names=["y_original_scale", "channel_contribution_original_scale"],
            extend_idata=False,
            progressbar=False,
            random_seed=rng
            )

        fig, ax = plt.subplots(figsize=(8, 4))

        for i, hdi_prob in enumerate([0.94, 0.5]):
            az.plot_hdi(
                x=mmm.model.coords["date"],
                y=(mmm.idata["posterior_predictive"]["y_original_scale"]),
                color="C0",
                smooth=False,
                hdi_prob=hdi_prob,
                fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob:.0%} HDI"},
                ax=ax,
            )

            az.plot_hdi(
                x=X_test[date_column],
                y=y_pred_test["y_original_scale"].unstack().transpose(..., "date"),
                color="C1",
                smooth=False,
                hdi_prob=hdi_prob,
                fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob:.0%} HDI"},
                ax=ax,
            )


        ax.plot(X_test[date_column], y_test, color="black")

        ax.plot(
            mmm.model.coords["date"],
            mmm.idata["posterior_predictive"]["y_original_scale"].mean(dim=("chain", "draw")),
            color="C0",
            linewidth=3,
            label="Posterior Predictive Mean",
        )

        ax.plot(
            X_test[date_column],
            y_pred_test["y_original_scale"].mean(dim=("sample")),
            color="C1",
            linewidth=3,
            label="Posterior Predictive Mean",
        )

        ax.axvline(X_test[date_column].iloc[0], color="C2", linestyle="--")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
        ax.set(ylabel="sales")
        ax.set_title("In-Sample and Out-of-Sample Predictions", fontsize=16, fontweight="bold")

        return fig, ax
        
    config = st.session_state.var_config
    time_col, target_col = get_k_by_v(config, "datetime"), get_k_by_v(config, "target")
    media_vars = [k for k, v in config.items() if v == 'media']
    prior_sigma = st.session_state.prior_sigma
    train_df = st.session_state.train_df
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    mmm = st.session_state['model']
    mmm.sample_posterior_predictive(X=X_train, random_seed=rng)

    cols = st.columns(2)
    with cols[0]:
        st.write("Model Trace")
        _ = az.plot_trace(
            data=mmm.fit_result,
            var_names=[
                "intercept_contribution",
                "y_sigma",
                "saturation_beta",
                "saturation_lam",
                "adstock_alpha",
                "gamma_control",
                "gamma_fourier",
            ],
            compact=True,
            backend_kwargs={"figsize": (12, 10), "layout": "constrained"},
        )
        fig = plt.gcf()
        fig.suptitle("Model Trace", fontsize=18, fontweight="bold")
        st.pyplot(fig, dpi=150)

    with cols[1]:
        st.write("Posterior Predictive Checks")
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, hdi_prob in enumerate([0.94, 0.5]):
            az.plot_hdi(
                x=mmm.model.coords["date"],
                y=(mmm.idata["posterior_predictive"].y_original_scale),
                color="C0",
                smooth=False,
                hdi_prob=hdi_prob,
                fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob:.0%} HDI"},
                ax=ax,
            )

        sns.lineplot(
            data=train_df, x=time_col, y=target_col, color="black", label="Sales", ax=ax
        )
        ax.legend(loc="upper left")
        ax.set(xlabel="date", ylabel="sales")
        ax.set_title("Posterior Predictive Checks", fontsize=16, fontweight="bold")
        st.pyplot(fig, dpi=150)
        st.caption("图1：后验预测检查")

    cols = st.columns(2)
    with cols[0]:
        st.write("Posterior Predictive - Channel Contributions")
        fig, ax = plot_channel_contributions(mmm, train_df, time_col, target_col)
        st.pyplot(fig, dpi=150)
        st.caption("图2：后验预测的渠道贡献分布")

    with cols[1]:
        st.write("Waterfall of components decomposition")
        fig, ax = mmm.plot.waterfall_components_decomposition(figsize=(10, 8))
        st.pyplot(fig, dpi=150)

    cols = st.columns(2)
    with cols[0]:
        st.write("Posterior Channel Contribution Share")
        channel_contribution_share = (
            mmm.idata["posterior"]["channel_contribution_original_scale"].sum(dim="date")
        ) / mmm.idata["posterior"]["channel_contribution_original_scale"].sum(
            dim=("date", "channel")
        )

        # Custom plot for demonstration
        fig, ax = plt.subplots(figsize=(10, 8))
        az.plot_forest(channel_contribution_share, combined=True, ax=ax)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        fig.suptitle("Posterior Channel Contribution Share", fontsize=18, fontweight="bold")
        st.pyplot(fig, dpi=150)

    with cols[1]:
        st.write("Return on Ads Spend (ROAS)")
        roas = mmm.incrementality.contribution_over_spend(frequency="all_time").rename("roas")
        fig, ax = plt.subplots(figsize=(10, 6))
        az.plot_forest(roas, combined=True, ax=ax)
        fig.suptitle("Return on Ads Spend (ROAS)", fontsize=16, fontweight="bold")
        st.pyplot(fig, dpi=150)

    cols = st.columns(2)
    with cols[0]:
        st.write("Channel Contribution Share vs ROAS")
        fig, ax = plot_channel_contribution_share(channel_contribution_share, media_vars, prior_sigma)
        st.pyplot(fig, dpi=150)
    with cols[1]:
        st.write("In-Sample and Out-of-Sample Predictions")
        fig, ax = plot_predictions(mmm, X_test, y_test, time_col)
        st.pyplot(fig, dpi=150)

# =========================
# Step 5: Budget Optimization
# =========================
if step == 'Step 5 – Budget Optimization' and 'model' in st.session_state:
    st.header('Step 5: Budget Optimization')
    st.write("Coming Soon...")
