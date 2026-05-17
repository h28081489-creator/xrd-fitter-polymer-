"""
拟合结果可视化。matplotlib 实现，兼容 Streamlit 与文件保存。
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，适合 Web/批量
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_fit_result(fit_result: Dict,
                    xc_result: Optional[Dict] = None,
                    title: str = "",
                    show_components: bool = True,
                    figsize=(10, 7)) -> plt.Figure:
    """
    绘制拟合结果：原始数据 / 总拟合 / 分量 / 残差。

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    tt = fit_result["two_theta"]
    y = fit_result["intensity"]
    yfit = fit_result["best_fit"]
    residual = y - yfit

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax)

    # 主图
    ax.plot(tt, y, "o", ms=3, color="black", label="Observed", alpha=0.6)
    ax.plot(tt, yfit, "-", color="red", lw=2, label="Total fit")

    if show_components:
        baseline = fit_result.get("baseline")
        if baseline is not None:
            ax.plot(tt, baseline, "--", color="gray", lw=1, label="Baseline")

        cryst_total = fit_result.get("crystalline_total")
        amorph_total = fit_result.get("amorphous_total")

        # 各结晶峰单独画（细线）
        components = fit_result.get("components", {})
        for pk in fit_result.get("peak_info", []):
            prefix = pk["prefix"]
            comp = components.get(prefix, None)
            if comp is None:
                continue
            if pk["kind"] == "crystalline":
                ax.fill_between(tt, 0, comp, alpha=0.15, color="tab:blue")
                ax.plot(tt, comp, "-", color="tab:blue", lw=0.8, alpha=0.7)
            else:
                ax.fill_between(tt, 0, comp, alpha=0.20, color="tab:orange")
                ax.plot(tt, comp, "-", color="tab:orange", lw=1.0, alpha=0.8)

        # 图例代理
        from matplotlib.patches import Patch
        legend_handles = [
            plt.Line2D([0], [0], marker='o', ls='', color='black', label='Observed', alpha=0.6, ms=4),
            plt.Line2D([0], [0], color='red', lw=2, label='Total fit'),
            Patch(facecolor='tab:blue', alpha=0.4, label='Crystalline peaks'),
            Patch(facecolor='tab:orange', alpha=0.4, label='Amorphous peak'),
        ]
        if baseline is not None:
            legend_handles.append(plt.Line2D([0], [0], color='gray', ls='--', label='Baseline'))
        ax.legend(handles=legend_handles, loc="best", fontsize=9)
    else:
        ax.legend(loc="best", fontsize=9)

    # 标题与 Xc
    title_parts = []
    if title:
        title_parts.append(title)
    if xc_result is not None and not np.isnan(xc_result["Xc_percent"]):
        title_parts.append(f"Xc = {xc_result['Xc_percent']:.2f}%")
    r2 = fit_result.get("r_squared", np.nan)
    if not np.isnan(r2):
        title_parts.append(f"R² = {r2:.4f}")
    ax.set_title("  |  ".join(title_parts), fontsize=11)

    ax.set_ylabel("Intensity (a.u.)")
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.3)

    # 残差图
    ax_r.plot(tt, residual, "-", color="purple", lw=0.8)
    ax_r.axhline(0, color="black", lw=0.5)
    ax_r.set_xlabel(r"2$\theta$ (°)")
    ax_r.set_ylabel("Residual")
    ax_r.grid(alpha=0.3)

    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """matplotlib Figure 转 PNG bytes，用于下载/打包。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def plot_batch_summary(summary_df,
                       peak_tracking_df=None,
                       figsize=(14, 10)) -> plt.Figure:
    """
    批量结果汇总图：
      A: Xc 柱状图
      B: R² 柱状图（R² < 0.95 标红）
      C: 关键峰中心 vs 样品
      D: 关键峰 FWHM vs 样品

    Parameters
    ----------
    summary_df : pd.DataFrame
        必须包含列：sample, Xc_percent, R_squared
    peak_tracking_df : pd.DataFrame or None
        长格式，列：sample, peak_label, center, fwhm
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    (axA, axB), (axC, axD) = axes

    samples = summary_df["sample"].tolist()
    x_pos = np.arange(len(samples))

    # A: Xc
    xc_vals = summary_df["Xc_percent"].to_numpy()
    axA.bar(x_pos, xc_vals, color="tab:blue", alpha=0.8)
    axA.set_xticks(x_pos)
    axA.set_xticklabels(samples, rotation=45, ha="right", fontsize=8)
    axA.set_ylabel("Xc (%)")
    axA.set_title("Crystallinity by sample")
    axA.grid(alpha=0.3, axis="y")
    for i, v in enumerate(xc_vals):
        if not np.isnan(v):
            axA.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)

    # B: R²
    r2_vals = summary_df["R_squared"].to_numpy()
    colors = ["red" if (not np.isnan(r) and r < 0.95) else "tab:green" for r in r2_vals]
    axB.bar(x_pos, r2_vals, color=colors, alpha=0.8)
    axB.axhline(0.95, color="black", ls="--", lw=0.8, label="R²=0.95")
    axB.set_xticks(x_pos)
    axB.set_xticklabels(samples, rotation=45, ha="right", fontsize=8)
    axB.set_ylabel("R²")
    axB.set_title("Fit quality (red: R² < 0.95)")
    axB.set_ylim(min(0.8, np.nanmin(r2_vals) - 0.05) if len(r2_vals) else 0.8, 1.01)
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3, axis="y")

    # C, D: 峰追踪
    if peak_tracking_df is not None and len(peak_tracking_df) > 0:
        labels = peak_tracking_df["peak_label"].unique()
        cmap = plt.get_cmap("tab10")
        for j, lbl in enumerate(labels):
            sub = peak_tracking_df[peak_tracking_df["peak_label"] == lbl]
            sub = sub.sort_values("sample_index")
            axC.plot(sub["sample_index"], sub["center"], "o-",
                     color=cmap(j % 10), label=lbl, ms=5)
            axD.plot(sub["sample_index"], sub["fwhm"], "s-",
                     color=cmap(j % 10), label=lbl, ms=5)
        axC.set_xticks(x_pos)
        axC.set_xticklabels(samples, rotation=45, ha="right", fontsize=8)
        axD.set_xticks(x_pos)
        axD.set_xticklabels(samples, rotation=45, ha="right", fontsize=8)
        axC.legend(fontsize=7, loc="best", ncol=2)
        axD.legend(fontsize=7, loc="best", ncol=2)
    else:
        for ax_ in (axC, axD):
            ax_.text(0.5, 0.5, "No peak tracking data",
                     ha="center", va="center", transform=ax_.transAxes)

    axC.set_ylabel("Peak center 2θ (°)")
    axC.set_title("Peak position trend")
    axC.grid(alpha=0.3)

    axD.set_ylabel("FWHM (°)")
    axD.set_title("Peak width (FWHM) trend")
    axD.grid(alpha=0.3)

    fig.tight_layout()
    return fig
