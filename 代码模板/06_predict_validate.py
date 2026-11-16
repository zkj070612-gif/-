"""C 题核心套路: 预测模型不值钱, "证明你的预测可信" 才值钱。
流程: 基线模型 -> 你的模型 -> 按时间顺序做交叉验证 -> 报误差和预测区间 -> 解释哪些特征重要。
依赖: numpy pandas scikit-learn matplotlib"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from plot_style import use_paper_style, save, PALETTE

# ---------- 1. 造数据 (比赛时换成 pd.read_csv) ----------
rng = np.random.default_rng(1)
n = 300
df = pd.DataFrame({"t": np.arange(n)})
df["x1"] = rng.normal(size=n)
df["x2"] = rng.normal(size=n)
df["noise_feature"] = rng.normal(size=n)                     # 故意放一个没用的特征
df["y"] = 0.05 * df.t + 3 * np.sin(df.t / 12) + 2 * df.x1 + df.x1 * df.x2 + rng.normal(0, 0.8, n)

# ---------- 2. 特征工程: 只能用"过去"的信息, 用了未来信息叫数据泄漏, 是 C 题最常见的硬伤 ----------
df["y_lag1"] = df.y.shift(1)
df["y_lag2"] = df.y.shift(2)
df["y_roll7"] = df.y.shift(1).rolling(7).mean()
df = df.dropna().reset_index(drop=True)
features = ["t", "x1", "x2", "noise_feature", "y_lag1", "y_lag2", "y_roll7"]
X, y = df[features].values, df.y.values

# ---------- 3. 时间序列交叉验证: 永远用前面训练、后面测试, 不能随机打乱 ----------
models = {
    "Naive (y_t = y_{t-1})": None,                            # 基线: 打不过它说明模型没价值
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=0),
    "GradientBoosting": GradientBoostingRegressor(random_state=0),
}
tscv = TimeSeriesSplit(n_splits=5)
print(f"{'model':<24}{'MAE mean':>10}{'MAE std':>10}")
for name, m in models.items():
    maes = []
    for tr, te in tscv.split(X):
        pred = df.y_lag1.values[te] if m is None else m.fit(X[tr], y[tr]).predict(X[te])
        maes.append(mean_absolute_error(y[te], pred))
    print(f"{name:<24}{np.mean(maes):>10.3f}{np.std(maes):>10.3f}")

# ---------- 4. 最终模型: 留最后 20% 做测试, 给出预测区间 ----------
split = int(len(df) * 0.8)
gb = lambda q: GradientBoostingRegressor(loss="quantile", alpha=q, random_state=0).fit(X[:split], y[:split])
lo, mid, hi = (gb(q).predict(X[split:]) for q in (0.05, 0.5, 0.95))
cover = np.mean((y[split:] >= lo) & (y[split:] <= hi))
print(f"test MAE = {mean_absolute_error(y[split:], mid):.3f}, 90% interval actually covers {cover:.0%} of test points")

# ---------- 5. 可解释性: 置换重要性 (在测试集上算, 比树模型自带的 feature_importances_ 可靠) ----------
rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X[:split], y[:split])
imp = permutation_importance(rf, X[split:], y[split:], n_repeats=20, random_state=0)
order = imp.importances_mean.argsort()
for i in order[::-1]:
    print(f"  {features[i]:<14} {imp.importances_mean[i]:+.3f} ± {imp.importances_std[i]:.3f}")

use_paper_style()
fig, ax = plt.subplots()
tt = df.t.values[split:]
ax.fill_between(tt, lo, hi, color=PALETTE[0], alpha=0.2, label="90% prediction interval")
ax.plot(tt, mid, color=PALETTE[0], label="Predicted (median)")
ax.scatter(tt, y[split:], s=10, color=PALETTE[1], label="Actual", zorder=3)
ax.set(xlabel="Time", ylabel="y", title="Out-of-sample forecast with uncertainty")
ax.legend()
save(fig, "fig_forecast")

# 读结果时注意 (这份示例数据是故意这么设计的):
# 1) 随机森林/梯度提升在交叉验证里没打赢 Naive 基线 —— 因为数据有上升趋势, 树模型不会外推。
#    "复杂模型 > 简单模型" 不是默认成立的, 不做基线对比就发现不了。修法: 先去趋势/差分, 或直接用线性模型。
# 2) 名义 90% 的预测区间在测试集上实际只覆盖了约 3/4 —— 区间偏窄。论文里要如实报这个覆盖率,
#    而不是只画一条漂亮的带子。能主动指出自己模型哪里不准的论文, 比假装完美的论文得分高。
