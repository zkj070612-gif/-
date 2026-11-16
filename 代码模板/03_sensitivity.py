"""灵敏度分析。评委明确要看 (官方规则里写的 testing: error analysis, sensitivity, stability)。
两种都要会:  (1) 局部 OAT: 每次只动一个参数 ±10%, 画龙卷风图
            (2) 全局: 所有参数同时随机抽样, 看输出分布 + 哪个参数最主导
把 model() 换成你自己的模型即可, 其余不用改。
依赖: numpy scipy matplotlib"""
import numpy as np
from scipy.stats import spearmanr, qmc
import matplotlib.pyplot as plt
from plot_style import use_paper_style, save, PALETTE

def model(p):
    """你的模型: 输入参数字典, 输出一个你关心的标量。这里用一个示例: 最终种群规模"""
    r, K, h, x0 = p["r"], p["K"], p["h"], p["x0"]
    x = x0
    for _ in range(50):                       # 带收获的 logistic 增长, 迭代 50 步
        x = max(x + r * x * (1 - x / K) - h * x, 0)
    return x

base = {"r": 0.30, "K": 1000.0, "h": 0.05, "x0": 50.0}
y_base = model(base)
print(f"baseline output = {y_base:.2f}")

# ---------- (1) 局部 OAT ----------
delta = 0.10
rows = []
for k in base:
    lo = model({**base, k: base[k] * (1 - delta)})
    hi = model({**base, k: base[k] * (1 + delta)})
    rows.append((k, (lo - y_base) / y_base * 100, (hi - y_base) / y_base * 100))
rows.sort(key=lambda r: max(abs(r[1]), abs(r[2])))
for k, lo, hi in rows[::-1]:
    print(f"  {k:>3}: -10% -> {lo:+6.2f}%   +10% -> {hi:+6.2f}%")

use_paper_style()
fig, ax = plt.subplots(figsize=(6, 3))
names = [r[0] for r in rows]
ax.barh(names, [r[1] for r in rows], color=PALETTE[0], label="parameter -10%")
ax.barh(names, [r[2] for r in rows], color=PALETTE[1], label="parameter +10%")
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="Change in output (%)", title="One-at-a-time sensitivity (tornado)")
ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
save(fig, "fig_sensitivity_tornado")

# ---------- (2) 全局: 拉丁超立方抽样 + Spearman 秩相关 ----------
n = 2000
keys = list(base)
u = qmc.LatinHypercube(d=len(keys), seed=0).random(n)        # [0,1) 上均匀
lo = np.array([base[k] * 0.8 for k in keys]); hi = np.array([base[k] * 1.2 for k in keys])
X = qmc.scale(u, lo, hi)                                      # 每个参数在 ±20% 内
Y = np.array([model(dict(zip(keys, row))) for row in X])
print(f"global: mean={Y.mean():.1f}, 5%-95% interval=[{np.percentile(Y,5):.1f}, {np.percentile(Y,95):.1f}]")
for j, k in enumerate(keys):
    rho = spearmanr(X[:, j], Y)[0]
    print(f"  Spearman rho({k}, output) = {rho:+.3f}")

fig, ax = plt.subplots()
ax.hist(Y, bins=40, color=PALETTE[0], alpha=0.85)
ax.axvline(y_base, color=PALETTE[1], ls="--", label="baseline")
ax.set(xlabel="Model output", ylabel="Count", title="Output distribution under ±20% parameter uncertainty")
ax.legend()
save(fig, "fig_sensitivity_global")
