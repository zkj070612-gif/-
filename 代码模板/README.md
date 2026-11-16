# 代码模板说明

这 6 个脚本补的是你已有的 `MCM-ICM-Algorithm` 仓库里**没有**的部分。全部在 Python 3 + numpy / scipy / pandas / matplotlib / scikit-learn / networkx 下实际跑通过。

安装依赖（一次即可）：

```
pip install numpy scipy pandas matplotlib scikit-learn networkx seaborn
```

运行：在本文件夹里打开终端，`python 01_ode_fit.py`。图会存成 PNG 和 PDF 两份。

| 文件 | 解决什么问题 | 主要对应题型 |
|---|---|---|
| `plot_style.py` | 全文统一的绘图风格，其他脚本都会 import 它 | 全部 |
| `01_ode_fit.py` | 微分方程建模、数值求解、用数据反推参数并给出误差 | A |
| `02_optimization.py` | 线性规划、整数规划、非线性规划、全局优化 | B、D、E |
| `03_sensitivity.py` | 局部（龙卷风图）和全局（拉丁超立方）灵敏度分析 | 全部，必做 |
| `04_monte_carlo.py` | 排队仿真、bootstrap 置信区间 | B、D |
| `05_graph_network.py` | 最短路、最小生成树、最大流、中心性、鲁棒性 | D |
| `06_predict_validate.py` | 预测模型的基线对比、时间序列交叉验证、预测区间、特征重要性 | C |

已有仓库里覆盖的内容（不重复写）：numpy / pandas 入门、各类统计图、数据预处理与插值、分类与回归、AHP / 熵权 / TOPSIS 等综合评价、LaTeX 模板。

## 怎么用才有效

不要收藏了就算学了。每个脚本按这个顺序过一遍：

1. 先跑通，看输出。
2. 合上文件，自己从空白文件重写一遍，卡住再看。
3. 改题：把 SIR 换成捕食者-猎物模型，把背包问题换成指派问题，把示例数据换成往年 C 题的真实数据。

第 2 步最难受，也最有用。比赛时没有时间现学语法。
