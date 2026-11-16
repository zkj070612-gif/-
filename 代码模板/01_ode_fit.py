"""A 题核心套路: 微分方程建模 -> 数值求解 -> 用数据反推参数 -> 画图。
例子: SIR 传染病模型。换成你的问题只需改 rhs() 和参数名。
依赖: numpy scipy matplotlib"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from plot_style import use_paper_style, save

def rhs(t, y, beta, gamma):
    """方程右端。y = [S, I, R], 返回 dy/dt"""
    S, I, R = y
    N = S + I + R
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]

def simulate(params, t_eval, y0):
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                    args=tuple(params), method="RK45", rtol=1e-8, atol=1e-8)
    return sol.y

# ---------- 1. 造一份"观测数据" (比赛时这里换成题目给的数据) ----------
rng = np.random.default_rng(0)
t = np.arange(0, 60)
y0 = [990, 10, 0]
true_params = (0.35, 0.10)
I_obs = simulate(true_params, t, y0)[1] * rng.normal(1, 0.05, t.size)   # 加 5% 噪声

# ---------- 2. 参数估计: 最小化 模型-数据 的残差 ----------
def residual(p):
    return simulate(p, t, y0)[1] - I_obs

fit = least_squares(residual, x0=[0.2, 0.2], bounds=([0, 0], [2, 2]))
beta, gamma = fit.x
# 参数的近似标准误 (论文里报 "beta = 0.35 ± 0.01" 比只报一个数强得多)
dof = t.size - fit.x.size
cov = np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun @ fit.fun) / dof
se = np.sqrt(np.diag(cov))
rmse = np.sqrt(np.mean(fit.fun ** 2))
print(f"beta  = {beta:.4f} ± {se[0]:.4f}   (true {true_params[0]})")
print(f"gamma = {gamma:.4f} ± {se[1]:.4f}   (true {true_params[1]})")
print(f"R0 = {beta/gamma:.2f},  RMSE = {rmse:.2f}")

# ---------- 3. 画图 ----------
use_paper_style()
S, I, R = simulate(fit.x, t, y0)
fig, ax = plt.subplots()
ax.scatter(t, I_obs, s=14, color="gray", label="Observed infected", zorder=3)
ax.plot(t, I, label="Fitted I(t)")
ax.plot(t, S, "--", label="S(t)")
ax.plot(t, R, "--", label="R(t)")
ax.set(xlabel="Time (days)", ylabel="Population", title="SIR model fitted to data")
ax.legend()
save(fig, "fig_ode_fit")
