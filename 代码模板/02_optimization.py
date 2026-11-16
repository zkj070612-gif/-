"""B/D/E 题核心套路: 优化。四种最常用情形各一个最小例子。
依赖: numpy scipy    (scipy>=1.9 自带 milp, 不用装别的求解器)"""
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds, minimize, differential_evolution

# ===== 1. 线性规划 LP: 生产计划 =====
# max 40x1 + 30x2   s.t. x1+x2<=40, 2x1+x2<=60, x>=0     (linprog 只会求 min, 所以目标取负)
res = linprog(c=[-40, -30], A_ub=[[1, 1], [2, 1]], b_ub=[40, 60], bounds=[(0, None)] * 2)
print("LP :", res.x.round(3), "profit =", -res.fun)
# 影子价格 = 约束右端每放宽 1 单位目标改善多少。写进论文就是现成的灵敏度分析
print("     shadow prices:", (-res.ineqlin.marginals).round(3))

# ===== 2. 整数规划 MILP: 0-1 背包 / 选址 / 指派 都是这个模板 =====
value = np.array([10, 13, 7, 8, 12]); weight = np.array([3, 4, 2, 3, 5]); cap = 9
res = milp(c=-value,
           constraints=LinearConstraint(weight[None, :], ub=cap),
           integrality=np.ones(5), bounds=Bounds(0, 1))
print("MILP: pick", np.flatnonzero(res.x.round() == 1), "value =", -res.fun)

# ===== 3. 带约束的非线性规划 NLP =====
# min (x-1)^2+(y-2.5)^2   s.t. x-2y+2>=0, x,y>=0
obj = lambda v: (v[0] - 1) ** 2 + (v[1] - 2.5) ** 2
cons = [{"type": "ineq", "fun": lambda v: v[0] - 2 * v[1] + 2}]     # ineq 表示 fun(v) >= 0
res = minimize(obj, x0=[2, 0], bounds=[(0, None)] * 2, constraints=cons, method="SLSQP")
print("NLP :", res.x.round(4), "f =", round(res.fun, 4))
# 注意: SLSQP 只保证局部最优。论文里要么证明问题是凸的, 要么多起点跑一遍证明结果稳定:
starts = np.random.default_rng(0).uniform(0, 5, (20, 2))
vals = [minimize(obj, s, bounds=[(0, None)] * 2, constraints=cons, method="SLSQP").fun for s in starts]
print("     multi-start spread:", round(max(vals) - min(vals), 8))

# ===== 4. 全局优化 (多峰、不可导、黑箱仿真): 差分进化 =====
# 比"遗传算法/粒子群"更省事, 而且是 scipy 自带的, 结果可复现
rastrigin = lambda v: 10 * len(v) + sum(v ** 2 - 10 * np.cos(2 * np.pi * v))
res = differential_evolution(rastrigin, bounds=[(-5.12, 5.12)] * 3, seed=0, tol=1e-8)
print("DE  :", res.x.round(4), "f =", round(res.fun, 6))

# 多目标怎么办: 加权和 w*f1+(1-w)*f2, 让 w 从 0 扫到 1, 把得到的点画成 Pareto 前沿。
