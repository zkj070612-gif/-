"""随机仿真两件套: (1) 离散事件仿真(排队) (2) bootstrap 给任意统计量配置信区间。
依赖: numpy"""
import numpy as np

# ===== 1. 单队列多服务台排队仿真 (收费站/登机/充电桩/医院 这类 B 题常用) =====
def simulate_queue(lam, mu, c, n_customers, rng):
    """到达率 lam, 单台服务率 mu, c 个服务台。返回每位顾客的等待时间"""
    arrivals = np.cumsum(rng.exponential(1 / lam, n_customers))
    service = rng.exponential(1 / mu, n_customers)
    free_at = np.zeros(c)                      # 每个服务台什么时候空出来
    waits = np.empty(n_customers)
    for i in range(n_customers):
        j = free_at.argmin()                   # 最早空闲的那台
        start = max(arrivals[i], free_at[j])
        waits[i] = start - arrivals[i]
        free_at[j] = start + service[i]
    return waits

rng = np.random.default_rng(42)                # 固定随机种子: 结果必须可复现
lam, mu = 0.9, 1.0
# 重复 30 次独立仿真, 报 均值 ± 标准误, 不要只跑一次就写结论
means = [simulate_queue(lam, mu, 1, 20000, rng)[2000:].mean() for _ in range(30)]   # 丢掉前 2000 个预热期
sim, se = np.mean(means), np.std(means, ddof=1) / np.sqrt(30)
theory = lam / (mu * (mu - lam))               # M/M/1 平均等待时间理论值, 用来验证仿真程序写对了
print(f"M/M/1 mean wait: simulated {sim:.3f} ± {se:.3f},  theory {theory:.3f}")
for c in (1, 2, 3):
    w = simulate_queue(lam, mu, c, 20000, rng)[2000:]
    print(f"  servers={c}: mean wait={w.mean():.3f}, 95th pct={np.percentile(w,95):.3f}")

# ===== 2. Bootstrap 置信区间: 对任何"从数据算出来的数"都能用 =====
data = rng.lognormal(mean=1.0, sigma=0.6, size=80)          # 换成你的样本
stat = np.median                                            # 换成你关心的统计量
boot = np.array([stat(rng.choice(data, data.size, replace=True)) for _ in range(5000)])
print(f"median = {stat(data):.3f}, 95% bootstrap CI = [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]")
