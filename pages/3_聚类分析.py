import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
from pathlib import Path
import yaml

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import calinski_harabasz_score

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
st.title('Clustering Analysis')

# Sidebar navigation
step = st.sidebar.radio(
    'Workflow',
    ['Step 0 – Data & Variables', 'Step 1 – Preprocessing', 'Step 2 – Variable Selection',
     'Step 3 – Clustering', 'Step 4 – Evaluation', 'Step 5 – Scoring']
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
    options = ['exclude', 'id', 'binary', 'nominal', 'ordinal', 'continuous']

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
        save_yaml(config, 'config/segmentation_variables.yaml')
        st.success('Variable configuration saved')

# =========================
# Step 1: Preprocessing
# =========================

if step == 'Step 1 – Preprocessing' and st.session_state.df is not None:
    st.header('Step 1: Preprocessing')
    config = st.session_state.var_config

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
        impute_methodO = st.selectbox('Imputation Method for Ordinal variables', ['mean', 'median', 'knn'])
    with col6:
        impute_method = st.selectbox('Imputation Method for Continues variables', ['mean', 'median', 'knn'])
    
    cont_vars = [k for k, v in config.items() if v == 'continuous']
    ord_vars = [k for k, v in config.items() if v == 'ordinal']
    nom_vars = [k for k, v in config.items() if v == 'nominal']
    bin_vars = [k for k, v in config.items() if v == 'binary']
    id_var = [k for k, v in config.items() if v == 'id']

    feat_vars = cont_vars + ord_vars + nom_vars + bin_vars
    df = st.session_state.df[id_var+feat_vars].copy()

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

    st.write('Dropped variables due to missing', drop_vars)

    # update features
    existing_cols = set(df.columns)
    cont_vars = [c for c in cont_vars if c in existing_cols]
    ord_vars  = [c for c in ord_vars  if c in existing_cols]
    nom_vars  = [c for c in nom_vars  if c in existing_cols]
    bin_vars  = [c for c in bin_vars  if c in existing_cols]

    feat_vars = cont_vars + ord_vars + nom_vars + bin_vars
    
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

        with open('config/preprocess_pipeline.yaml', 'w', encoding='utf-8') as f:
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

    howmany = st.slider('Number of variables to keep',
        math.ceil(df.shape[1] * 0.5), min(50, df.shape[1]), 
        math.ceil(df.shape[1] * 0.8)
        )

    variances = df.drop(columns=id_var, errors='ignore').var().sort_values(ascending=False)
    selected = variances.head(howmany).index.tolist()

    st.write('Selected variables')
    st.write(selected)

    st.session_state.selected_vars = selected
    df = df[id_var + selected].copy()
    st.session_state.processed_df = df

    st.write('The final dataset used for modeling')
    st.dataframe(df.head(), use_container_width=True)

# =========================
# Step 3: Clustering
# =========================

if step == 'Step 3 – Clustering' and 'selected_vars' in st.session_state:
    st.header('Step 3: K Selection & Clustering')
    df = st.session_state.processed_df
    X = df[st.session_state.selected_vars]

    mink, maxk = st.slider('K range', 2, 10, (3, 8))

    scores = {}
    for k in range(mink, maxk + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        scores[k] = calinski_harabasz_score(X, labels)

    score_df = pd.DataFrame.from_dict(scores, orient='index', columns=['CH Score'])
    st.line_chart(score_df)

    best_k = score_df['CH Score'].idxmax()
    st.success(f'Recommended K = {best_k}')

    final_k = st.number_input('Final K', value=int(best_k))
    km = KMeans(n_clusters=final_k, n_init=20, random_state=42)
    df['cluster'] = km.fit_predict(X)

    st.session_state.clustered_df = df
    st.session_state.kmeans = km

# =========================
# Step 4: Evaluation
# =========================

if step == 'Step 4 – Evaluation' and 'clustered_df' in st.session_state:
    st.header('Step 4: Evaluation')
    st.subheader('1. PCA and LDA charts')

    df = st.session_state.clustered_df

    # ---------------- 采样 ----------------
    max_points = 10000  # 最大绘图点数
    if df.shape[0] > max_points:
        df_plot = df.sample(n=max_points, random_state=42)
    else:
        df_plot = df.copy()

    X = df_plot[st.session_state.selected_vars]
    y = df_plot['cluster']

    # ---------------- PCA ----------------
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    pca_df = pd.DataFrame(Xp, columns=['PC1', 'PC2'])
    pca_df['cluster'] = y

    # ---------------- LDA ----------------
    lda = LinearDiscriminantAnalysis(n_components=2)
    Xl = lda.fit_transform(X, y)
    lda_cols = [f'LD{i+1}' for i in range(Xl.shape[1])]
    lda_df = pd.DataFrame(Xl, columns=lda_cols)
    lda_df['cluster'] = y

    # ---------------- 并排显示图表 ----------------
    col1, col_space, col2 = st.columns([5, 0.5, 5])  # 0.5 作为空隙

    with col1:
        st.markdown("PCA Scatter Plot")
        st.scatter_chart(pca_df, x='PC1', y='PC2', color='cluster')

    with col2:
        st.markdown("LDA Scatter Plot")
        st.scatter_chart(
            lda_df,
            x=lda_cols[0],
            y=lda_cols[1] if Xl.shape[1] > 1 else lda_cols[0],
            color='cluster'
        )

    # ========= Variable Summary by Cluster =========
    st.subheader('2. Variable Summary by Cluster')
    selected_vars = st.session_state.selected_vars  # 建模使用的特征列

    # ---------------- 聚类均值分析 ----------------
    cluster_means = df.groupby('cluster')[selected_vars].mean().reset_index()
    st.markdown("Cluster Means Table")
    st.dataframe(cluster_means, use_container_width=True)
    
    # ---------------- 平行坐标系图 ----------------
    st.markdown("Parallel Coordinates Plot")
    # Plotly 的 px.parallel_coordinates 需要列是数值型
    # height_slider = st.slider("Graph Height", 400, 1000, 600)
    fig = px.parallel_coordinates(
        cluster_means,
        color='cluster',
        dimensions=selected_vars,
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=cluster_means['cluster'].mean(),
        labels={c: c for c in selected_vars},  # 保留原始特征名
        # height=height_slider,
        height=400
    )
    # 设置每个维度的小数位数（例如保留2位小数）
    for i, col in enumerate(selected_vars):
        fig.data[0].dimensions[i].tickformat = '.2f'  # 保留2位小数
    # 调整字体和边距
    fig.update_layout(
        font=dict(size=18, 
        color="black"  # 明确设置字体颜色，但是在系统白色背景和黑色背景见会会出现异常
        ), 
        margin=dict(l=80, r=50, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Step 5: Scoring
# =========================

if step == 'Step 5 – Scoring' and 'kmeans' in st.session_state:
    st.header('Step 5: Scoring New Dataset')

    conn = get_conn()
    sources = pd.read_sql("SELECT * FROM data_sources", conn)

    # ========= 显式选择 + 提交 =========
    with st.form("scoring_form"):
        src = st.selectbox(
            "Select a dataset for prediction",
            sources["name"],
            index=None,
            placeholder="Select a dataset..."
        )
        submitted = st.form_submit_button("▶️ Run")

    # ========= 只在提交时运行评分 =========
    if submitted and src is not None:
        path = sources.loc[sources["name"] == src, "path"].values[0]

        try:
            config = st.session_state.var_config
            id_var = [k for k, v in config.items() if v == 'id']
            selected = st.session_state.selected_vars
            new_df = pd.read_csv(path, usecols=id_var + selected)
            st.write('Sample of new dataset')
            st.dataframe(new_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

        # 加载预处理 pipeline
        try:
            with open('config/preprocess_pipeline.yaml', 'r', encoding='utf-8') as f:
                pipeline = yaml.safe_load(f)
        except Exception as e:
            st.error(f"Failed to load preprocessing pipeline: {e}")
            st.stop()

        # ------------------- 名义变量 -------------------
        for col, params in pipeline.get('nominal', {}).items():
            if col in new_df.columns:
                enc = OrdinalEncoder(
                    categories=[params['categories']],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                new_df[[col]] = enc.fit_transform(new_df[[col]].astype(str))

        # ------------------- 有序变量 -------------------
        for col, params in pipeline.get('ordinal', {}).items():
            if col in new_df.columns:
                enc = OrdinalEncoder(
                    categories=[params['categories']],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                new_df[[col]] = enc.fit_transform(new_df[[col]].astype(str))

        # ------------------- 二值变量 -------------------
        for col, params in pipeline.get('binary', {}).items():
            if col in new_df.columns:
                true_vals = params['true_values']
                new_df[col] = new_df[col].apply(lambda x: 1 if str(x).lower() in true_vals else 0)

        # ------------------- 连续变量 -------------------
        for col, params in pipeline.get('continuous', {}).items():
            if col in new_df.columns:
                new_df[col].fillna(params['fill_value'], inplace=True)
                new_df[col] = (new_df[col] - params['mean']) / params['std']

        # 特征对齐（必须与建模时顺序一致）
        Xnew = new_df[st.session_state.selected_vars]
        labels = st.session_state.kmeans.predict(Xnew)
        new_df['cluster'] = labels

        # 保存结果到 session state
        st.session_state.scoring_result = new_df.copy()

    # ========= 关键防线：检查是否有结果 =========
    if 'scoring_result' not in st.session_state:
        st.info("Please select a dataset and click **Run** to score new data.")
        st.stop()

    # ========= 从这里开始，结果 100% 安全 =========
    result_df = st.session_state.scoring_result
    st.success(f"Scoring completed: {result_df.shape[0]} rows classified into clusters.")

    st.write("Sample of scoring result:")
    st.dataframe(result_df.head(), use_container_width=True)
    st.download_button(
                    "💾 下载聚类结果完整明细数据",
                    data=result_df.to_csv().encode('utf-8'),
                    file_name="Ignite platform-Segmentation Results.csv",
                    mime="csv")
