import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from services.db_service import get_conn
from typing import Union, Optional
from datetime import date
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


def detect_anomalies_stat(data, target='Stock_quantity', ds='datetime', threshold=5, 
    # weights_list=[1,1,1.1,1,1],
    weights_list=[2,1,1.5,2,3]
    ):
    """
    阈值               异常定义                    适用场景
    3.0        较敏感，可能包含轻度异常            预警、早期发现
    3.5        标准推荐值（Iglewicz & Hoaglin）   通用、平衡
    4.0～5.0   只抓极端异常值                     剔除明显错误、脏数据
    >5.0       极其严格，几乎只保留完美数据        高精度建模前清洗
    """
    result = data.copy()
    
    # 1. 检查输入数据
    if result.empty:
        raise ValueError("Input data is empty!")
    
    # 2. 确保数据类型正确
    result['date'] = pd.to_datetime(result[ds])
    result[target] = pd.to_numeric(result[target], errors='coerce')  # 非数值转为NaN
    result.dropna(subset=[target], inplace=True)  # 删除无效值
    
    # 3. 生成季度列
    result['quarter'] = result['date'].dt.to_period('Q')
    quarters = sorted(result['quarter'].unique())
    
    # 4. 遍历季度检测异常
    for i, quarter in enumerate(quarters):
        if i == 0:
            # 针对第一个季度的数据，就用自身数据作为历史数据
            past_quarters = quarters[0:1]
        else:
            # 获取过去5个季度的数据和当前季度已有数据
            past_quarters = quarters[max(0, i-5):i+1]

        past_data = result[result['quarter'].isin(past_quarters)]

        q_mean = np.mean(past_data[target])

        q1, q25, q50, q75, q99 = np.quantile(past_data[target], [0.01, 0.25, 0.5, 0.75, 0.99])
        iqr = q75 - q25
        BoxPlot_lower = q25 - iqr * 3
        BoxPlot_upper = q75 + iqr * 3
        
        # 调整权重计算加权median
        n_groups = len(past_data.groupby('quarter'))
        weights = np.array(weights_list[-n_groups:])
        weights = weights / weights.sum()
        
        # 计算加权统计量
        weighted_values = []
        for (q, group), w in zip(past_data.groupby('quarter')[target], weights):
            n_rep = max(1, int(round(w * 10)))  # 放大10倍后取整，保证精度
            weighted_values.extend(np.repeat(group, n_rep))
        
        weighted_values = np.array(weighted_values)

        # 计算统计指标
        weighted_median = np.median(weighted_values)
        weighted_mad = np.median(np.abs(weighted_values - weighted_median))

        # 标记当前季度的异常
        current_mask = result['quarter'] == quarter
        current_values = result.loc[current_mask, target]
        
        modified_z_scores = 0.6745 * (current_values - weighted_median) / max(weighted_mad, 1e-6)
        delta = (threshold * weighted_mad) / 0.6745
        mZScore_lower = weighted_median - delta
        mZScore_upper = weighted_median + delta
        modified_z_scores = modified_z_scores.fillna(0)
        
        # 更新结果列
        result.loc[current_mask, f'{target}_mean'] = q50
        result.loc[current_mask, f'{target}_median'] = q_mean
        result.loc[current_mask, f'{target}_q1'] =  q1
        result.loc[current_mask, f'{target}_q25'] = q25
        result.loc[current_mask, f'{target}_q75'] = q75
        result.loc[current_mask, f'{target}_q99'] = q99

        result.loc[current_mask, f'{target}_modified_z_score'] = modified_z_scores
        result.loc[current_mask, f'{target}_mZScore_lower'] = mZScore_lower
        result.loc[current_mask, f'{target}_mZScore_upper'] = mZScore_upper
        result.loc[current_mask, f'{target}_mz_anomaly'] = (np.abs(modified_z_scores) > threshold).astype(bool)

        result.loc[current_mask, f'{target}_BoxPlot_lower'] = BoxPlot_lower
        result.loc[current_mask, f'{target}_BoxPlot_upper'] = BoxPlot_upper
        result.loc[current_mask, f'{target}_box_anomaly'] = ((current_values < BoxPlot_lower) | (current_values > BoxPlot_upper)).astype(bool)

        result.loc[current_mask, f'{target}_trunc_anomaly'] = ((current_values < q1) | (current_values > q99)).astype(bool)

    return result

# Plotly 绘图函数
def plot_with_plotly(plot_data, target, method, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data[target], name='实际值', mode='lines', line=dict(color='blue')))
    if method == '修正Z_score':
        anomaly_col = f'{target}_mz_anomaly'
        fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data[f'{target}_mZScore_upper'], 
                                 name='上界', mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data[f'{target}_mZScore_lower'], 
                                 name='下界', fill='tonexty', mode='lines', line=dict(width=0), 
                                 fillcolor='rgba(0,100,80,0.2)'))
    elif method == "箱线图法":
        anomaly_col = f'{target}_box_anomaly'
        fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data[f'{target}_BoxPlot_upper'], 
                                 name='上界', mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data[f'{target}_BoxPlot_lower'], 
                                 name='下界', fill='tonexty', mode='lines', line=dict(width=0), 
                                 fillcolor='rgba(0,100,80,0.2)'))
    elif method == "高分位截断法":
        anomaly_col = f'{target}_trunc_anomaly'
        fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data[f'{target}_q99'], 
                                 name='上界', mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=plot_data['date'], y=plot_data[f'{target}_q1'], 
                                 name='下界', fill='tonexty', mode='lines', line=dict(width=0), 
                                 fillcolor='rgba(0,100,80,0.2)'))
    
    anomalies = plot_data[plot_data[anomaly_col]]
    n_anomalies = plot_data[anomaly_col].sum()
    title = f"{title}    |    检测到 {n_anomalies} 个异常点！"

    fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies[target], name='异常点', mode='markers', 
                             marker=dict(color='red', size=8)))
    fig.update_layout(title=title, xaxis_title='日期', yaxis_title=target, legend_title='图例', 
                      hovermode='x unified', template='plotly_white',
                      xaxis=dict(rangeslider=dict(visible=True), type="date")
                      )
    return fig

def get_parameters(key_prefix="", target_options=None, dims_options=None):
    """
    获取用户输入的参数，并为每个组件指定唯一的 key。
    """
    st.write("请输入参数进行分析：")
    # === 第一行：开始日期 + 结束日期（2列）===
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=pd.to_datetime("2023-01-01"), 
                                   key=f"{key_prefix}_start_date")
    with col2:
        end_date = st.date_input("结束日期", value=pd.to_datetime("2025-02-28"), 
                                 key=f"{key_prefix}_end_date")

    # === 第二行：分析级别 + 目标变量 + 异常检测方法（3列）===
    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.selectbox("选择目标变量", target_options, key=f"{key_prefix}_target")
    with col2:
        analysis_level = st.selectbox("选择分析级别", dims_options, key=f"{key_prefix}_analysis_level")
    with col3:
        method = st.selectbox("选择异常检测方法", ["修正Z_score", "箱线图法", "高分位截断法"], 
                              key=f"{key_prefix}_method")
    
    return start_date, end_date, analysis_level, target, method

def get_k_by_v(dict, value, default=None):
    """返回字典中第一个匹配值的键，未找到返回 default"""
    return next((k for k, v in dict.items() if v == value), default)

# 提取数据分析部分
def analyze_data(data, analysis_level, target, config, start_date, end_date, key_prefix=""):
    """
    根据分析级别过滤数据，并生成标题。
    """
    ds, l1, l2, l3 = get_k_by_v(config, "datetime"), get_k_by_v(config, "level-1"), get_k_by_v(config, "level-2"), get_k_by_v(config, "level-3")

    data[ds] = pd.to_datetime(data[ds])

    if analysis_level == l3:
        l3_v = st.selectbox(f"选择 {l3}", data[l3].unique(), 
            key=f"{key_prefix}_sku_select"  # 添加唯一 key
        )
        filtered_data = data[data[l3] == l3_v]
        filtered_data = detect_anomalies_stat(filtered_data, target=target, ds=ds)
        title = f"{l3}: {l3_v}"
    elif analysis_level == l1:
        l1_v = st.selectbox(f"选择{l1}", data[l1].unique(), 
            key=f"{key_prefix}_category_select"  # 添加唯一 key
        )
        filtered_data = data[data[l1] == l1_v].groupby(ds).agg({
            target: 'sum'
        }).reset_index()
        filtered_data = detect_anomalies_stat(filtered_data, target=target, ds=ds)
        title = f"{l1}: {l1_v}"
    else:
        l2_v = st.selectbox(f"选择{l2}", data[l2].unique(), 
            key=f"{key_prefix}_subcategory_select"  # 添加唯一 key
        )
        filtered_data = data[data[l2] == l2_v].groupby(ds).agg({
            target: 'sum'
        }).reset_index()
        filtered_data = detect_anomalies_stat(filtered_data, target=target, ds=ds)
        title = f"{l2}: {l2_v}"

    st.write("请在上部选择参数并点击 '运行分析' 按钮")

    # 过滤日期范围
    filtered_data = filtered_data[
        (filtered_data[ds] >= pd.to_datetime(start_date)) &
        (filtered_data[ds] <= pd.to_datetime(end_date))
    ]
    return filtered_data, title

def prepare_ts_data(data, sku=None, category=None, subcategory=None, target="sales_volume", 
                     start_date=None, end_date=None):
    """
    使用SARIMAX模型检测异常并返回预测结果
    返回：DataFrame（含预测数据与异常标记）及图表标题
    """
    if sku is not None:
        full_data = data[data['sku'] == sku][['date', target]].rename(columns={'date': 'ds', target: 'y'})
        title = f'{target} for SKU: {sku}'
    elif category is not None:
        full_data = data[data['category'] == category].groupby('date')[target].sum().reset_index().rename(
            columns={'date': 'ds', target: 'y'})
        title = f'{target} for Category: {category}'
    elif subcategory is not None:
        full_data = data[data['subcategory'] == subcategory].groupby('date')[target].sum().reset_index().rename(
            columns={'date': 'ds', target: 'y'})
        title = f'{target} for Subcategory: {subcategory}'
    else:
        raise ValueError("必须指定 sku、category 或 subcategory")

    # 将ds列转换为datetime
    full_data['ds'] = pd.to_datetime(full_data['ds'])

    # 时间范围过滤
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        result_df = full_data[full_data['ds'] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        result_df = result_df[result_df['ds'] <= end_date]
    
    return result_df, title

# ARIMAX
def detect_anomalies_sarimax(data, confidence_coef=0.95):
    """
    使用SARIMAX模型检测异常并返回预测结果
    返回：DataFrame（含预测数据与异常标记）及图表标题
    """

    # 将ds设置为索引
    data.set_index('ds', inplace=True)
    
    # 重新采样以确保时间序列连续（按天）
    full_data = data.asfreq('D')
    
    # 填充缺失值
    full_data['y'] = full_data['y'].fillna(method='ffill').fillna(method='bfill')
    
    # 确定SARIMAX参数
    # 季节周期设为365天（年季节性）
    seasonal_order = (1, 1, 1, 7)  # P, D, Q, s (周季节性)
    
    # 拟合SARIMAX模型
    # 使用差分来处理趋势
    model = SARIMAX(full_data['y'], 
                    order=(1, 1, 1),  # p, d, q
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    fitted_model = model.fit(disp=False)
    
    # 预测未来30天
    forecast_steps = 30
    forecast_result = fitted_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_conf_int = forecast_result.conf_int(alpha=1-confidence_coef)
    
    # 预测历史数据以获取置信区间
    history_forecast = fitted_model.get_prediction(start=0, end=len(full_data)-1)
    history_mean = history_forecast.predicted_mean
    history_conf_int = history_forecast.conf_int(alpha=1-confidence_coef)
    
    # 创建结果DataFrame
    # 首先为历史数据创建DataFrame
    history_df = pd.DataFrame({
        'ds': full_data.index,
        'y': full_data['y'],
        'yhat': history_mean.values,
        'yhat_lower': history_conf_int.iloc[:, 0].values,
        'yhat_upper': history_conf_int.iloc[:, 1].values
    })
    
    # 创建未来预测的DataFrame
    last_date = full_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    future_df = pd.DataFrame({
        'ds': future_dates,
        'y': [np.nan] * forecast_steps,  # 未来的真实值未知
        'yhat': forecast_mean.values,
        'yhat_lower': forecast_conf_int.iloc[:, 0].values,
        'yhat_upper': forecast_conf_int.iloc[:, 1].values
    })
    
    # 合并历史和未来数据
    result_df = pd.concat([history_df, future_df], ignore_index=True)
    
    # 添加异常标记
    result_df['is_anomaly'] = False
    # 只对有实际值的数据点进行异常检测
    mask = ~result_df['y'].isna()
    result_df.loc[mask, 'is_anomaly'] = (
        (result_df.loc[mask, 'y'] < result_df.loc[mask, 'yhat_lower']) | 
        (result_df.loc[mask, 'y'] > result_df.loc[mask, 'yhat_upper'])
    )
    
    # 重置索引
    result_df.reset_index(drop=True, inplace=True)
    
    return result_df

def plot_with_plotly2(plot_data, title, target, confidence_coef):
    """
    使用 Plotly 绘制交互式时间序列图
    返回 Plotly Figure 对象
    """
    fig = go.Figure()

    # 实际值
    fig.add_trace(go.Scatter(x=plot_data['ds'], y=plot_data['y'], name='实际值', mode='lines+markers', line=dict(color='blue')))
    # 预测值
    fig.add_trace(go.Scatter(x=plot_data['ds'], y=plot_data['yhat'], name='预测值', mode='lines', line=dict(color='orange')))
    # 置信区间
    fig.add_trace(go.Scatter(x=plot_data['ds'], y=plot_data['yhat_upper'], name=f'{confidence_coef*100:.0f}% 置信区间上界',
                             mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=plot_data['ds'], y=plot_data['yhat_lower'], name=f'{confidence_coef*100:.0f}% 置信区间',
                             fill='tonexty', mode='lines', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)'))
    # 异常点
    anomalies = plot_data[plot_data['is_anomaly']]
    n_anomalies = plot_data['is_anomaly'].sum()
    title = f"{title}    |    检测到 {n_anomalies} 个异常点！"

    fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], name='异常点', mode='markers', marker=dict(color='red', size=8)))

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title=target,
        legend_title='图例',
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    return fig

def get_parameters2(key_prefix="", target_options=None):
    """
    获取用户输入参数，并为每个组件指定唯一 key
    """
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=pd.to_datetime("2024-01-01"), key=f"{key_prefix}_start_date")
        target = st.selectbox("选择目标变量", target_options, key=f"{key_prefix}_target")
    with col2:
        end_date = st.date_input("结束日期", value=pd.to_datetime("2025-02-28"), key=f"{key_prefix}_end_date")
        analysis_level = st.selectbox("选择分析级别", ["SKU", "Category", "Subcategory"], key=f"{key_prefix}_analysis_level")
        
    return start_date, end_date, target, analysis_level

def analyze_data2(data, analysis_level, key_prefix=""):
    """
    根据分析级别过滤数据，并生成标题
    """
    if analysis_level == "SKU":
        sku = st.selectbox("选择 SKU", data['sku'].unique(), key=f"{key_prefix}_sku_select")
        category, subcategory = None, None
    elif analysis_level == "Category":
        category = st.selectbox("选择品类", data['category'].unique(), key=f"{key_prefix}_category_select")
        sku, subcategory = None, None
    else:
        subcategory = st.selectbox("选择子类", data['subcategory'].unique(), key=f"{key_prefix}_subcategory_select")
        sku, category = None, None
    return sku, category, subcategory

def should_refresh_results(cache_key, current_params):
    """检查参数是否发生变化，决定是否需要刷新结果"""
    param_key = f"{cache_key}_params"
    
    # 如果参数未保存过，或参数发生变化，则需要刷新
    if param_key not in st.session_state:
        st.session_state[param_key] = current_params
        return True
    
    if st.session_state[param_key] != current_params:
        st.session_state[param_key] = current_params
        return True
    
    return False

def run_analysis_auto(data, sku, category, subcategory, target, start_date, end_date, confidence_coef, cache_key):
    """自动运行分析并检测参数变化"""
    # 生成当前参数签名
    current_params = {
        'sku': sku, 'category': category, 'subcategory': subcategory,
        'target': target, 
        'start_date': start_date, 'end_date': end_date,
        'confidence_coef': confidence_coef
    }
    
    # 检查是否需要刷新
    if should_refresh_results(cache_key, current_params) or cache_key not in st.session_state:
        with st.spinner("分析中..."):
            full_data = detect_anomalies_sarimax(data, confidence_coef=confidence_coef)
            st.session_state[cache_key] = {"full_data": full_data}
    
    return st.session_state[cache_key]["full_data"]

def show_result(full_data, title, target, confidence_coef, radio_key):
    """
    图表只画一次，radio 只控制表格
    """
    # ===== 1. 缓存图（只算一次）=====
    fig_key = f"{radio_key}_fig"
    if fig_key not in st.session_state:
        st.session_state[fig_key] = plot_with_plotly2(
            full_data, title, target, confidence_coef
        )

    st.plotly_chart(st.session_state[fig_key], use_container_width=True)

    # ===== 2. radio 只影响 dataframe =====
    option = st.radio(
        "选择要显示的数据类型：",
        ('全部数据', '异常数据', '正常数据'),
        horizontal=True,
        key=radio_key
    )

    if option == '全部数据':
        show_df = full_data
    elif option == '异常数据':
        show_df = full_data[full_data['is_anomaly']]
    else:
        show_df = full_data[~full_data['is_anomaly']]

    st.write(f"当前显示 {show_df.shape[0]} 条记录")
    st.dataframe(show_df, use_container_width=True)


st.set_page_config(layout="wide")

st.header('📉Abnormal Analysis')
st.subheader('📥1. Load Data')

# ========= 选择数据 =========
conn = get_conn()
sources = pd.read_sql("SELECT * FROM data_sources", conn)

with st.form("data_select_form"):
    src = st.selectbox(
        "Select a dataset for modeling",
        sources["name"],
        index=None,
        placeholder="Select a dataset..."
    )
    submitted = st.form_submit_button("▶️ Load Data and Analyse")

# ========= 读取数据 =========
if submitted:
    path = sources.loc[sources["name"] == src, "path"].values[0]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    st.session_state.df = df

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


st.subheader('🔢2. Variable Profiling & Classification')
st.warning("datetime, level-1, level-2, level-3 等类型只能各选择一个字段，target 可选择多个字段！")

sample_n = 3
col_widths = [1/2, 2, 1, 1*sample_n, 1, 1, 2]
header_cols = st.columns(col_widths)
header_texts = ["#", "Field Name", "Type", f"Sample ({sample_n})", "Missing %", "Unique", "Classification"]

for col, text in zip(header_cols, header_texts):
    col.markdown(f"**{text}**")

st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)  # 表头分隔线

prev_config = st.session_state.get('var_config', {})
options = ['exclude', 'datetime', 'target', 'level-1', 'level-2', 'level-3']

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
        if 'datetime' in c.lower() or 'date' in c.lower():
            default_value = 'datetime'
        elif 'target' in c.lower():
            default_value = 'target'
        elif c_type =="datetime64[ns]":
            default_value = 'datetime'
        elif c_type =="object":
            default_value = 'level-1'
        elif c_type in ["int64","float64"]:
            default_value = 'target'

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

st.session_state.var_config = config

# 删除无效字段
drop_vars = [k for k, v in config.items() if v == 'exclude']
df = df.drop(columns=drop_vars)

st.markdown("")  # 空一行

st.subheader('📉3. Abnormal Analysis')
tab1, tab2 = st.tabs(["🧮 统计方法", "🧬 机器学习"])
with tab1:
    target_options = [k for k, v in config.items() if v == 'target']
    dims_options = [k for k, v in config.items() if v in ["level-1", "level-2", "level-3"]]
    start_date, end_date, analysis_level, target, method = get_parameters(key_prefix="sell_in", 
        target_options=target_options, dims_options=dims_options)
    filtered_data, title = analyze_data(df, analysis_level, target, config, start_date, end_date, key_prefix="sell_in")
    # 运行分析
    if st.button("运行分析", key="stat"):
        plot_data = filtered_data.sort_values(by="date")
        fig = plot_with_plotly(plot_data, target, method, f"{title}    |    Target: {target}     |    Method: {method}")
        st.plotly_chart(fig, use_container_width=True)
        st.write("异常检测结果（含辅助指标）：")
        st.dataframe(plot_data[(plot_data[f'{target}_mz_anomaly'] ==True) | (plot_data[f'{target}_box_anomaly'] ==True)])
with tab2:
    sell_in_target_options = [k for k, v in config.items() if v == 'target']
    start_date, end_date, target, analysis_level = get_parameters2(key_prefix="sell", 
        target_options=sell_in_target_options)
    
    rename_dict = {'入账日期': 'date', '业务品牌': 'brand', '总品类': 'category', '品类': 'subcategory', '产品标准名称': 'sku'}
    df.rename(columns=rename_dict, inplace=True)
    sku, category, subcategory = analyze_data2(df, analysis_level, key_prefix="sell_")
    confidence_coef = st.slider("置信度", 0.8, 0.99, 0.95, key=f"ML_cc")

    # 运行分析
    if st.button("运行分析", key="ml"):
        try:
            full_data, title = prepare_ts_data(df, sku, category, subcategory, target, start_date, end_date)
            full_data = run_analysis_auto(full_data, sku, category, subcategory, target, start_date, end_date, confidence_coef, cache_key="raw_data")
            show_result(full_data, title, target, confidence_coef, radio_key="sell_in_radio")
        except Exception as e:
            st.error(f"分析过程中出现错误: {e}")
            st.info("如果看到Prophet相关错误，请先在命令行中执行修复步骤。")
