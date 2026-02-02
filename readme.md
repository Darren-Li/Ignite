项目路径：C:\Users\lwq07\Aha\Ignite

启动命令：
在项目路径下启动终端
1. 启动python虚拟环境
	conda activate prophet_env
2. 启动项目
	cd Aha\Ignite
	streamlit run Ignite.py


C:\Users\lwq07\Aha\Ignite\db\app.db


项目目录
streamlit_anomaly_app/
│
├─ app.py                      # Streamlit 主入口
├─ requirements.txt            # 依赖
│
├─ data/
│   ├─ raw/                    # 原数据（标签：原数据）
<!-- │   └─ analysis/                # 分析结果数据（标签：分析结果） -->
│   └─ profiling               # data profiling 
│
├─ db/
│   └─ app.db                  # Sqlite 数据库
│
├─ services/
│   ├─ db_service.py           # 数据库连接 & 元数据管理
│   ├─ data_loader.py          # 文件 / DB 数据加载
│
├─ pages/
│   ├─ 1_数据管理.py
│   └─ 2_异常分析.py
│   └─ 3_聚类分析.py
│   └─ 4_预测分析.py
│   └─ 5_营销组合优化.py
│
├─ config/
│   ├─ 
│   └─ 
│
├─ models/
│   ├─ 
│   └─ 
│
├─ outputs/
│   ├─ 
│   └─ 
│
└─ utils/
    └─ helpers.py              # 公共工具函数
