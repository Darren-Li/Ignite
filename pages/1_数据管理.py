import streamlit as st
import pandas as pd
import json
from services.db_service import get_conn
from services.data_loader import load_file, load_db
from pathlib import Path
from sqlalchemy import create_engine
from ydata_profiling import ProfileReport
from datetime import datetime
import secrets

st.set_page_config(layout="wide") # 必须是该页面脚本中的第一个 Streamlit 调用


def sample_df(df: pd.DataFrame, frac=0.3, min_rows=1000, max_rows=5000, random_state=42):
    """
    对 DataFrame 进行采样：
    - 默认抽取 frac 的比例，但不少于 min_rows，不多于 max_rows
    """
    n_total = len(df)
    n_sample = min(max(int(n_total * frac), min_rows), max_rows, n_total)
    
    # 随机采样
    df_sampled = df.sample(n=n_sample, random_state=random_state)
    return df_sampled

def generate_profile(df, data_name):
    """
    为给定的 DataFrame 生成交互式数据质量分析报告（HTML 格式），并保存到指定目录。
    """
    profiling_dir = Path("data/profiling")
    profiling_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    file_path = profiling_dir / f"{data_name}.html"
    df_sampled = sample_df(df)
    profile = ProfileReport(df_sampled, title="Ignite Data Profiling", 
        minimal=True,      # True - 禁用耗时计算（如相关性、样本、重复行检测等）报告生成更快 适合大数据集或快速预览
        explorative=False, # 启用更深入的探索（如文本分析、URL/Email 检测等），比 minimal=False 更细
        sensitive=False,   # 若为 True，会尝试检测敏感信息（如身份证、银行卡号），但可能误报
        progress_bar=False)
    profile.to_file(file_path)
    return file_path


st.header("📁 数据管理")
st.subheader("🔗 数据源连接")
tab1, tab2 = st.tabs(["📂 上传文件", "🗄️ 连接数据库"])

# ---------------- 上传文件 ----------------
with tab1:
    uploaded = st.file_uploader("上传 Excel / CSV", type=["csv", "xlsx"])
    if uploaded:
        df = load_file(uploaded)
        upload_id = secrets.token_urlsafe(6)  # 生成 8 字符的短 ID（约 48 位熵，冲突概率极低） 注意：参数是字节数，不是字符数！
        safe_filename = f"{Path(uploaded.name).stem}_{upload_id}.csv"  # 将数据保存为csv格式
        save_path = Path("data/raw") / safe_filename
        df.to_csv(save_path, index=False)
        conn = get_conn()
        conn.execute(
            """INSERT INTO data_sources 
               (name, source_type, path, tag, upload_time) 
               VALUES (?, ?, ?, ?, ?)""",
            (uploaded.name, "file", str(save_path), "原数据", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
        conn.commit()
        st.success("文件数据源已添加")

# ---------------- 数据库 ----------------
with tab2:
    db_type = st.selectbox("数据库类型", ["sqlite", "mysql", "postgresql"])

    # 固定前缀
    prefix_map = {
        "sqlite": "sqlite:///",
        "mysql": "mysql+pymysql://",
        "postgresql": "postgresql+psycopg2://"
    }
    fixed_prefix = prefix_map[db_type]

    # 用户输入剩余部分
    user_input = st.text_input(
        "数据库信息/文件路径：💡SQLite: C:/path/to/db.db  |  MySQL/PostgreSQL: user:pass@host:port/dbname",
        placeholder="SQLite: C:/path/to/db.db | MySQL/PostgreSQL: user:pass@host:port/dbname [按回车键加载数据]"
    )

    # 拼接完整连接字符串
    conn_str = fixed_prefix + user_input if user_input else None

    # 初始化表列表
    tables = []
    selected_table = None

    # 连接数据库获取表列表
    if conn_str:
        try:
            engine = create_engine(conn_str)
            # SQLAlchemy 2.x 使用 inspect
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            if tables:
                st.success(f"找到 {len(tables)} 张表")
                selected_table = st.selectbox("选择表", tables)
            else:
                st.warning("数据库中没有表，请检查")
        except Exception as e:
            st.error(f"连接失败: {e}")

    # 导入选中表
    if selected_table and st.button("导入表数据"):
        try:
            df = pd.read_sql(f"SELECT * FROM {selected_table}", engine)
            st.success(f"成功导入表: {selected_table}，行数: {len(df)}, 列数: {len(df.columns)}")

            # 同时写入你的 data_sources 表
            conn = get_conn()
            conn.execute(
                """INSERT INTO data_sources 
                   (name, source_type, path, tag, upload_time) 
                   VALUES (?, ?, ?, ?, ?)""",
                (selected_table, db_type, conn_str, "原数据", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()

        except Exception as e:
            st.error(f"导入失败: {e}")


st.markdown("")  # 空一行

# ---------------- 数据源列表 ----------------
st.subheader("📋 数据源列表")
st.markdown('<hr style="margin:8px 0; border-top: 1px solid #eee">', unsafe_allow_html=True)

conn = get_conn()
df_sources = pd.read_sql("SELECT * FROM data_sources ORDER BY id DESC", conn)

if df_sources.empty:
    st.info("还没有添加数据源")
else:    
    # 表头（保持单行）
    col_widths_header = [0.5, 2, 1, 1, 1, 1]
    header_cols = st.columns(col_widths_header)
    header_texts = ["#", "File name", "File type", "Label", "Upload time", "Action"]
    for col, text in zip(header_cols, header_texts):
        col.markdown(f"**{text}**")

    st.markdown('<hr style="margin:1px 0; border-top: 2px solid #eee">', unsafe_allow_html=True)

    # 遍历每条数据源
    for idx, row in df_sources.iterrows():
        # ===== 第一行：# | 文件名 | 类型 | 标签 | 上传时间 | 操作 =====
        data_profiling_fn = f"{Path(row['name']).stem}_{row['id']}"
        row1_cols = st.columns(col_widths_header)  # 保持跟表头一样的列数和宽度
        with row1_cols[0]:
            st.write(idx + 1)
        with row1_cols[1]:
            st.write(row['name'])
        with row1_cols[2]:
            st.markdown(f"`{row['source_type']}`")
        with row1_cols[3]:
            st.write(row['tag'] if pd.notna(row['tag']) else "—")
        with row1_cols[4]:
            st.write(row['upload_time'])
        with row1_cols[5]:
            if st.button("删除", key=f"del_{row['id']}"):
                # 在数据库中删除数据
                conn.execute("DELETE FROM data_sources WHERE id=?", (row["id"],))
                conn.commit()
                # 删除保存的数据源文件
                raw_file = Path(row['path'])
                if raw_file.exists():
                    raw_file.unlink()
                # 删除data profiling file
                html_file = Path(f"data/profiling/{data_profiling_fn}.html")
                if html_file.exists():
                    html_file.unlink()
                st.rerun()

        # ===== 读取数据 =====
        try:
            if row["source_type"] == "file":
                df_file = pd.read_csv(row["path"])
            else:
                df_file = load_db(row["source_type"], row["path"], row["name"])
        except Exception as e:
            st.error(f"加载失败: {e}")

        # ===== 第二行：查看数据样本 =====
        with st.expander("👁️ View the sample(5)"):
            st.dataframe(df_file.head(), use_container_width=True)

        # ===== 第三行：汇总分析 =====
        with st.expander("📊 Data profiling"):
            html_file = Path(f"data/profiling/{data_profiling_fn}.html")
            if df_file.empty:
                st.warning("No data in the file/table!")
            elif not html_file.exists():
                st.warning("报告尚未生成，正在生成中...")
                html_file = generate_profile(df_file, data_profiling_fn)
                html_content = html_file.read_text(encoding="utf-8")
                st.components.v1.html(html_content, height=700, scrolling=True)
                st.download_button(
                    "💾 下载 HTML 报告",
                    data=html_content,
                    file_name=html_file.name,
                    mime="text/html")
            else:
                html_content = html_file.read_text(encoding="utf-8")
                st.components.v1.html(html_content, height=700, scrolling=True)
                st.download_button(
                    "💾 下载 HTML 报告",
                    data=html_content,
                    file_name=html_file.name,
                    mime="text/html")

        # 每条数据后加分隔线
        st.markdown('<hr style="margin:1px 0; border-top: 1px solid #ddd">', unsafe_allow_html=True)
