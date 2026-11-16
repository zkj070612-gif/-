"""论文统一绘图风格。每个脚本开头:  from plot_style import use_paper_style, save
规则: 全文图用同一套字体/配色; 图里只写英文; 一律存 300dpi PNG + PDF(矢量, LaTeX 用)。"""
import matplotlib
matplotlib.use("Agg")          # 不弹窗, 直接存文件; 想交互看图就删掉这行
import matplotlib.pyplot as plt

PALETTE = ["#2E5EAA", "#E07A3F", "#3A9D6E", "#B5446E", "#7D6BB0", "#8C8C8C"]

def use_paper_style():
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.0), "figure.dpi": 110, "savefig.dpi": 300,
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "lines.linewidth": 1.8, "legend.frameon": False,
        "savefig.bbox": "tight",
    })

def save(fig, name):
    """同时存 PNG 和 PDF。name 不带扩展名, 例如 save(fig, 'fig1_sir_fit')"""
    fig.savefig(f"{name}.png")
    fig.savefig(f"{name}.pdf")
    plt.close(fig)
