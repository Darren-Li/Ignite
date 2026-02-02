import streamlit as st

st.set_page_config(layout='wide')
st.title("Ignite数据分析平台")

# 分组
feature_groups = {
  "数据准备与治理": [
    {
      "title": "数据管理",
      "icon": "📂",
      "url": "/数据管理",
      "description": "上传本地文件、连接数据库表；管理数据源；数据质量分析 -缺失值、异常值、重复值、分布偏移检测"
    },
    # {
    #   "title": "数据质量分析(coming soon)",
    #   "icon": "🧹",
    #   "url": "/数据质量",
    #   "description": "缺失值、异常值、重复值、分布偏移检测"
    # },
    {
      "title": "特征处理中心(coming soon)",
      "icon": "🧱",
      "url": "/特征工程",
      "description": "标准化、分箱、编码、时间序列特征构造"
    }
  ],

  "诊断与洞察分析": [
    {
      "title": "异常分析",
      "icon": "⚠️",
      "url": "/异常分析",
      "description": "业务指标异常识别与定位（Z-Score / IQR / 时间序列异常）"
    },
    {
      "title": "指标波动归因(coming soon)",
      "icon": "🧭",
      "url": "/指标归因",
      "description": "拆解指标变化原因，定位关键影响因素"
    },
    {
      "title": "客群洞察",
      "icon": "👥",
      "url": "/聚类分析",
      "description": "客户分群与画像分析（K-means / PCA）"
    }
  ],

  "预测分析": [
    {
      "title": "趋势与需求预测(coming soon)",
      "icon": "📈",
      "url": "/趋势预测",
      "description": "时间序列预测（ARIMA / SARIMAX / Prophet）"
    },
    {
      "title": "预测分析（回归/分类）",
      "icon": "📊",
      "url": "/预测分析",
      "description": "销售额、需求量等连续目标预测（Linear / Random Forest）；转化、流失、响应概率预测（Logistic / XGBoost）"
    },
    # {
    #   "title": "行为预测（分类）",
    #   "icon": "🔍",
    #   "url": "/预测分类",
    #   "description": "转化、流失、响应概率预测（Logistic / XGBoost）"
    # },
    {
      "title": "库存与补货预测(coming soon)",
      "icon": "📦",
      "url": "/库存预测",
      "description": "需求预测 + 安全库存测算（生命周期 / 波动性）"
    }
  ],

  "优化与策略分析": [
    {
      "title": "营销组合优化",
      "icon": "💹",
      "url": "/营销组合优化",
      "description": "营销效果评估，营销预算分配优化"
    },
    {
      "title": "价格与促销策略分析(coming soon)",
      "icon": "🏷️",
      "url": "/价格策略",
      "description": "价格弹性分析、促销机制效果评估"
    },
    {
      "title": "资源配置优化(coming soon)",
      "icon": "⚙️",
      "url": "/资源优化",
      "description": "在约束条件下寻找最优资源配置方案"
    }
  ],

  "模拟与决策支持": [
    {
      "title": "情景模拟分析(coming soon)",
      "icon": "🎯",
      "url": "/情景模拟",
      "description": "不同业务策略下的结果对比与影响评估"
    },
    {
      "title": "策略对比与历史回测(coming soon)",
      "icon": "🔄",
      "url": "/策略回测",
      "description": "基于历史数据回测不同策略的真实效果"
    }
  ]
}

cols_per_row = 3  # 每行卡片数量，可调整
# 渲染分组
for group_name, features in feature_groups.items():
    st.markdown(f"### {group_name}")
    with st.container():
        for i in range(0, len(features), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, feature in zip(cols, features[i:i+cols_per_row]):
                with col:
                    st.markdown(f"""
                                <div style="
                                    border:1px solid #ddd; 
                                    border-radius:10px; 
                                    padding:15px; 
                                    display:flex; 
                                    align-items:flex-start; 
                                    height:160px; 
                                    background-color:#f9f9f9;
                                    overflow:hidden;
                                    ">
                                    <div style="font-size:40px; margin-right:15px;">{feature['icon']}</div>
                                    <div style="flex:1; display:flex; flex-direction:column; justify-content:space-between;">
                                        <h4 style="margin:0;">
                                            <a href="{feature['url']}" style="text-decoration:none; color:#000;">
                                                {feature['title']}
                                            </a>
                                        </h4>
                                        <p style="margin:0; font-size:16px; color:#555; overflow-y:auto;">{feature['description']}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            st.markdown("")

st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: gray; font-size: 16px;">
        Copyright©2026 南京秉智数据科技有限公司
    </div>
    """,
    unsafe_allow_html=True
)