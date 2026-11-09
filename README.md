# OEC_final

低轨卫星边缘计算（Orbital Edge Computing）方向的论文与仿真代码合集，内容对应 WCNC 2026 论文 *Real-Time Pollution Detection for IIoT: A Hybrid Reinforcement Learning Offloading Framework for Multi-Tier Satellite Networks*。

## 目录结构

```
OEC_final/
├── paper/                       论文与建模文档
│   ├── 中期报告——陈苗苗.docx          硕士学位论文中期报告（当前版本）
│   ├── 面向遥感任务…开题报告…docx     开题报告
│   ├── a122-miaomiao final.pdf       WCNC 2026 论文终稿
│   ├── RQ1-问题建模-精简版.md         点一：任务/模型/切分决策的形式化建模
│   ├── 中期报告-研究内容结构.md        报告章节结构与写作口径
│   ├── figs/                        报告插图（图 2-3/2-4/2-5 为文献原图，待重绘）
│   └── archive/                     报告历史版本快照（本地保留，不入库）
├── Reference/                   参考文献 PDF（第三方出版物，本地保留，不入库）
├── 格式参考/                    格式参考报告（含个人信息，本地保留，不入库）
├── simulator/                   仿真与强化学习代码（git submodule）
└── .gitmodules
```

标注“不入库”的目录已在 `.gitignore` 中忽略：仓库只保留当前版本的报告与插图，历史快照、参考文献与格式模板留在本地。

`simulator/` 指向独立仓库 `git@github.com:LOVEYOUTHREETHOUND/OEC-RL-Simulator.git`，保留其自身的提交历史。修改代码后需在该目录内单独提交、推送。

## 拉取

```bash
git clone --recurse-submodules git@github.com:LOVEYOUTHREETHOUND/OEC_final.git
```

已克隆但子模块为空时：

```bash
git submodule update --init --recursive
```

## 运行仿真代码

```bash
cd simulator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

主要入口在 `simulator/scripts/`：

- `train_a2c.py`、`train_ppo.py`、`train_hrl_a2c.py`：基线/分层强化学习训练
- `train_dt.py`、`collect_trajectories_for_dt.py`：Decision Transformer 数据采集与训练
- `eval_agent.py`、`evaluate.py`：策略评估
- `test_env.py`、`test_env_interaction.py`：环境自检

环境与智能体参数在 `simulator/configs/`（`satellites.yaml`、`trainning.yaml`、`environment/`、`agent/`）。
