import streamlit as st
from services.db_service import init_db

# 设置页面配置
st.set_page_config(
    page_title="数据分析平台",          # 标签页标题
    page_icon="📊",                    # 标签页图标
    layout="wide",                     # 布局方式
    initial_sidebar_state="expanded",  # 侧边栏状态
    menu_items={
        'Get Help': 'https://docs.streamlit.io',
        'Report a bug': "https://github.com/streamlit/streamlit/issues",
        'About': "# 数据分析平台 v1.0"
    }
)

init_db()

st.header("📊Ignite数据分析平台")
st.markdown("")
st.subheader("在这里！数据分析简单、直白、又能轻松上手！")
st.markdown("")

sidebar_style = """
<style>
.sidebar-quote {
    font-size: 24px;
    color: #555555;
    font-style: italic;
    margin-left: 30%;
    margin-bottom: 21px;
}
</style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)

quotes = [
    "在数据的海洋里，洞察是你唯一的指南针。",
    "分析不只是看数字，而是理解背后的故事。",
    "让数据告诉你真相，而不是让假设支配你。",
    "数据是原材料，分析是工艺，洞察是产品。",
    "统计不会骗人，选择性解释才会。"
]

for quote in quotes:
    st.markdown(f'<div class="sidebar-quote">{quote}</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: gray; font-size: 16px;">
        Copyright©2026 南京秉智数据科技有限公司
    </div>
    """,
    unsafe_allow_html=True
)