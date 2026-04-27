import streamlit as st

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="Ignite 数据分析平台",
    layout="wide"
)

# =========================
# Sidebar
# =========================
st.sidebar.title("Ignite")

menu = [
    "首页",
    "数据准备与治理",
    "诊断与归因分析",
    "预测分析",
    "优化与策略分析",
    "模拟与决策支持",
    "因果与实验",
    "个性化推荐",
]

selected_menu = st.sidebar.radio("导航", menu)

# =========================
# 推荐路径
# =========================
def show_recommend_paths():
    st.markdown("## 👍 推荐分析路径")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
📉 **销售下滑分析**
- 异常分析 → 指标归因 → 营销组合优化

📈 **用户增长**
- 客群洞察 → Uplift → 推荐系统
""")

    with col2:
        st.info("""
🎯 **提升转化**
- 预测模型 → 下一步最佳行动

🔁 **提升留存**
- 生存分析 → 流失预测 → 精准触达
""")

# =========================
# 卡片组件（稳定版）
# =========================
def render_card(features, cols_per_row=3):
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
                    height:180px; 
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
                  </div>""", unsafe_allow_html=True)
            st.markdown("")

# =========================
# 模块定义（产品级）
# =========================
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

  "诊断与归因分析": [
    {
      "title": "异常分析",
      "icon": "⚠️",
      "url": "/异常分析",
      "description": "业务指标异常识别与定位（Z-Score /修正Z-Score/ IQR / 时间序列预测算法）"
    },
    {
      "title": "客群洞察",
      "icon": "👥",
      "url": "/聚类分析",
      "description": "客户分群与画像分析（K-means / PCA）"
    },
    {
      "title": "指标波动归因(coming soon)",
      "icon": "🧭",
      "url": "/指标归因",
      "description": "拆解指标变化原因，定位关键影响因素"
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
      "description": "销售额、需求量等连续目标预测（Linear / Random Forest）；转化、流失、响应倾向性分类目标预测（Logistic / XGBoost）"
    },
    {
      "title": "生存分析",
      "icon": "⏳",
      "url": "/生存分析",
      "description": "“用户多久后会复购/流失”、“何时发券效果最好” 等，不仅能预测“是否发生”，还能预测“何时发生”"
    },
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
  ],

  "因果与实验": [
    {
      "title": "A/B测试(coming soon)",
      "icon": "🧪",
      "url": "/AB测试",
      "description": "A/B测试"
    },
    {
      "title": "因果推断",
      "icon": "🔗",
      "url": "/因果推断",
      "description": "因果推断"
    },
  ],

  "个性化推荐": [
    {
      "title": "个性化推荐(coming soon)",
      "icon": "🎯",
      "url": "/个性化推荐",
      "description": "商品/内容推荐"
    },
  ],
}

# =========================
# 主页面
# =========================
st.title("Ignite 数据分析平台")

if selected_menu == "首页":
    show_recommend_paths()
    st.markdown("## 🧩 功能模块")
    for group_name, features in feature_groups.items():
        st.markdown(f"### {group_name}")
        render_card(features, cols_per_row=3)
else:
    st.markdown(f"### {selected_menu}")
    features = feature_groups.get(selected_menu, [])
    render_card(features, cols_per_row=3)

# =========================
# Footer
# =========================
st.markdown(
  """
  <div style="text-align: center; margin-top: 50px; color: gray; font-size: 16px;">
      Copyright©2026 南京秉智数据科技有限公司
  </div>
  """,
  unsafe_allow_html=True
)
