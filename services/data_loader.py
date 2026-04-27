import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st
from services.db_service import get_conn


def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def load_db(db_type, conn_str, table):
    engine = create_engine(conn_str)
    return pd.read_sql(f"SELECT * FROM {table}", engine)

def load_data(key="Ignite"):
    conn = get_conn()
    sources = pd.read_sql("SELECT * FROM data_sources", conn)

    with st.form("load_form"):
        src = st.selectbox("Select a dataset for modeling", sources["name"], index=None, placeholder="Select a dataset...", key=key)
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
    return df

def var_profiling(df, 
    options = ['exclude', 'id', 'target', 'target_time', 'target_event', 'binary', 'nominal', 'ordinal', 'continuous']
    ):
    sample_n = 3
    col_widths = [1/2, 2, 1, 1*sample_n, 1, 1, 2]
    header_cols = st.columns(col_widths)
    header_texts = ["#", "Field Name", "Type", f"Sample ({sample_n})", "Missing %", "Unique", "Classification"]

    for col, text in zip(header_cols, header_texts):
        col.markdown(f"**{text}**")

    st.markdown('<hr style="margin:1px 0">', unsafe_allow_html=True)  # 表头分隔线

    prev_config = st.session_state.get('var_config', {})

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
            elif 'event' == c.lower() or '_event' in c.lower():
                default_value = 'target_event'
            elif 'conversion' == c.lower() or 'conversion_' in c.lower():
                default_value = 'target'
            elif 'treatment' == c.lower() or '_treatment' in c.lower():
                default_value = 'treatment'
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
    return config