"""
XRD 结晶度分析工具 —— 主入口
针对 PTT/PBT 共混体系（也适用于其他半结晶聚合物）。

启动：
    streamlit run app.py
"""

import streamlit as st


# ============================================================
# 页面全局配置（必须在任何 st.* 调用之前）
# ============================================================
st.set_page_config(
    page_title="XRD 结晶度分析工具",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**XRD 结晶度分析工具**\n\n"
            "基于 lmfit 的多峰分离与结晶度（Xc）计算。\n\n"
            "- 单文件模式：交互式调参\n"
            "- 批量模式：统一主峰表 + 一键处理\n\n"
            "适用于 PTT、PBT 及其共混体系。"
        )
    },
)


# ============================================================
# 延迟导入模式模块（让页面配置先生效）
# ============================================================
from modes import single as single_mode  # noqa: E402
from modes import batch as batch_mode    # noqa: E402


# ============================================================
# 侧边栏导航
# ============================================================
def sidebar_nav():
    st.sidebar.title("🔬 XRD Xc 工具")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "选择工作模式",
        options=["单文件分析", "批量分析"],
        index=0,
        key="app_mode_radio",
        help=(
            "**单文件**：交互式调参、寻峰、峰表编辑、查看拟合细节\n\n"
            "**批量**：多文件统一主峰表，一键处理，导出汇总报表"
        ),
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("📖 使用流程", expanded=False):
        st.markdown(
            "**单文件模式**\n"
            "1. 上传 CSV\n"
            "2. 设置 2θ 范围、峰型、基线\n"
            "3. 自动寻峰或手动添加峰\n"
            "4. 标记每个峰为结晶/无定形\n"
            "5. 拟合 → 查看 Xc → 下载\n\n"
            "**批量模式**\n"
            "1. 上传所有 CSV\n"
            "2. 选一个参考样品生成主峰表\n"
            "3. 编辑主峰表（label/kind/center）\n"
            "4. 一键批量拟合\n"
            "5. 下载汇总 Excel 或完整 ZIP"
        )

    with st.sidebar.expander("⚙️ 默认参数来源", expanded=False):
        st.markdown(
            "默认峰位、容差、FWHM 范围等定义在：\n\n"
            "`config/materials.yaml`\n\n"
            "可直接编辑该文件以适应不同体系。"
        )

    with st.sidebar.expander("ℹ️ 关于 Xc 计算", expanded=False):
        st.markdown(
            "结晶度公式：\n\n"
            "$$X_c = \\frac{A_{cryst}}{A_{cryst} + A_{amorph}}$$\n\n"
            "其中 A 为各峰拟合后的**积分面积**"
            "（lmfit 中的 amplitude 参数即代表积分面积）。\n\n"
            "对 PTT/PBT 共混体系，本工具只输出**总结晶度**，"
            "不做 PTT/PBT 各自结晶度的分离（峰严重重叠，分离不可靠）。"
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("v0.1.0  |  lmfit + Streamlit")

    return mode


# ============================================================
# 主流程
# ============================================================
def main():
    mode = sidebar_nav()

    if mode == "单文件分析":
        single_mode.render()
    elif mode == "批量分析":
        batch_mode.render()
    else:
        st.error(f"未知模式：{mode}")


if __name__ == "__main__":
    main()
