import streamlit as st
from services.db_service import init_db

# 设置页面配置
st.set_page_config(layout="wide")

st.title("📊 Ignite数据分析平台")
st.markdown("""
**功能说明**
- 数据管理：
	- 上传本地文件 / 连接数据库表
- 异常分析：
	- Z-Score / Box-plot / SARIMAX
- 聚类分析：
    - 聚类算法：K-means
    - 特征填充、转化、筛选
    - 聚类结果分析
    - 新数据预测
- 预测分析：
    - 回归分析：
        - 算法：Linear Regression
        - 特征填充、转化、筛选
        - 预测结果分析
        - 新数据预测
    - 分类分析：
        - 算法：Logistic Regression
        - 特征填充、转化、筛选
        - 预测结果分析
        - 新数据预测
- MMM(Marketing Mix Modeling)：
    - 算法：MMM
    - 特征处理：
        - Ad-stcok
        - Saturation
    - 预测结果分析：
        - Efficiency
        - Effectiveness
        - Saturation Curves
    - 新数据预测
    - 营销预算优化
""")

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: gray; font-size: 16px;">
        Copyright©2026 南京秉智数据科技有限公司
    </div>
    """,
    unsafe_allow_html=True
)