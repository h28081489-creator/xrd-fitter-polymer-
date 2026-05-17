"""
单文件分析模式：交互式 UI。
工作流：
  1. 上传 CSV
  2. 设置拟合范围、峰型、基线
  3. 自动寻峰 + 手动增删/编辑峰
  4. 标记每个峰为"结晶峰"或"无定形峰"
  5. 执行拟合
  6. 查看 Xc、R²、可视化、下载结果
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

from core.data_io import load_xrd_csv, crop_range
from core.peak_detection import auto_find_with_hints, load_materials_config
from core.fitting import build_and_fit
from core.crystallinity import compute_crystallinity
from core.plotting import plot_fit_result, fig_to_png_bytes


# ============================================================
# Session State 初始化
# ============================================================
def _init_state():
    defaults = {
        "single_data_loaded": False,
        "single_two_theta_full": None,
        "single_intensity_full": None,
        "single_meta": None,
        "single_peaks_df": None,       # 当前峰表（DataFrame，可编辑）
        "single_fit_result": None,
        "single_xc_result": None,
        "single_fit_settings": None,   # 拟合时使用的设置快照
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# 峰表 ↔ DataFrame 互转
# ============================================================
def _peaks_to_df(peaks_list, default_kind="crystalline"):
    """[{'center','height','fwhm_est','hint'}, ...] -> DataFrame"""
    rows = []
    for p in peaks_list:
        rows.append({
            "use": True,
            "kind": default_kind,
            "center": round(float(p["center"]), 4),
            "fwhm_est": round(float(p.get("fwhm_est", 0.3)), 4),
            "height": round(float(p.get("height", 0.0)), 2),
            "hint": p.get("hint", ""),
        })
    return pd.DataFrame(rows, columns=["use", "kind", "center", "fwhm_est", "height", "hint"])


def _df_to_peak_lists(df):
    """DataFrame -> (crystalline_peaks, amorphous_peaks)"""
    cryst, amorph = [], []
    if df is None or len(df) == 0:
        return cryst, amorph
    for _, row in df.iterrows():
        if not bool(row.get("use", True)):
            continue
        try:
            entry = {
                "center": float(row["center"]),
                "fwhm_est": float(row["fwhm_est"]),
                "height": float(row["height"]),
            }
        except (ValueError, TypeError):
            continue
        kind = str(row.get("kind", "crystalline")).strip().lower()
        if kind.startswith("amorph"):
            amorph.append(entry)
        else:
            cryst.append(entry)
    return cryst, amorph


# ============================================================
# 主渲染函数
# ============================================================
def render():
    _init_state()
    st.header("📄 单文件分析模式")

    cfg = load_materials_config("config/materials.yaml")
    fit_defaults = cfg.get("fitting_defaults", {})
    cryst_cfg = cfg.get("crystalline_peak", {})
    amorph_cfg = cfg.get("amorphous_peak", {})

    # ============ 1. 文件上传 ============
    st.subheader("1️⃣ 上传 XRD 数据")
    uploaded = st.file_uploader("选择 CSV 文件（两列：2θ, Intensity）",
                                 type=["csv", "txt", "xy", "dat"],
                                 key="single_uploader")

    if uploaded is not None:
        # 仅当文件名变化时重新加载
        if (not st.session_state.single_data_loaded
                or st.session_state.single_meta is None
                or st.session_state.single_meta.get("filename") != uploaded.name):
            try:
                tt, inten, meta = load_xrd_csv(uploaded.getvalue(), filename=uploaded.name)
                st.session_state.single_two_theta_full = tt
                st.session_state.single_intensity_full = inten
                st.session_state.single_meta = meta
                st.session_state.single_data_loaded = True
                st.session_state.single_peaks_df = None
                st.session_state.single_fit_result = None
                st.session_state.single_xc_result = None
                st.success(f"✅ 已加载 {uploaded.name}（{meta['n_points']} 个数据点，"
                           f"2θ ∈ [{meta['two_theta_min']:.2f}, {meta['two_theta_max']:.2f}]）")
            except Exception as e:
                st.error(f"❌ 加载失败：{type(e).__name__}: {e}")
                return

    if not st.session_state.single_data_loaded:
        st.info("请先上传一个 CSV 文件以继续。")
        return

    tt_full = st.session_state.single_two_theta_full
    inten_full = st.session_state.single_intensity_full
    meta = st.session_state.single_meta

    with st.expander("📋 文件元信息", expanded=False):
        st.json(meta)

    # ============ 2. 拟合设置 ============
    st.subheader("2️⃣ 拟合设置")
    col1, col2, col3 = st.columns(3)
    with col1:
        # 数据实际范围
        data_tt_min = float(meta["two_theta_min"])
        data_tt_max = float(meta["two_theta_max"])

        # 配置中的期望默认值，钳制到数据实际区间内
        cfg_tt_min = float(fit_defaults.get("two_theta_min", 10.0))
        cfg_tt_max = float(fit_defaults.get("two_theta_max", 35.0))
        default_tt_min = min(max(cfg_tt_min, data_tt_min), data_tt_max)
        default_tt_max = min(max(cfg_tt_max, data_tt_min), data_tt_max)
        # 保证上限严格大于下限，避免 number_input 初始就违反 tt_max > tt_min
        if default_tt_max <= default_tt_min:
            default_tt_max = data_tt_max

        tt_min = st.number_input("2θ 下限 (°)",
                                  value=default_tt_min,
                                  min_value=data_tt_min,
                                  max_value=data_tt_max,
                                  step=0.5)
        tt_max = st.number_input("2θ 上限 (°)",
                                  value=default_tt_max,
                                  min_value=data_tt_min,
                                  max_value=data_tt_max,
                                  step=0.5)
    with col2:
        peak_shape = st.selectbox("峰型",
                                   ["PseudoVoigt", "Gaussian", "Lorentzian", "Voigt"],
                                   index=["PseudoVoigt", "Gaussian", "Lorentzian", "Voigt"].index(
                                       fit_defaults.get("peak_shape", "PseudoVoigt")))
        baseline_kind = st.selectbox("基线", ["linear", "polynomial", "none"], index=0)
    with col3:
        baseline_degree = st.number_input("多项式阶数（基线为 polynomial 时）",
                                            value=2, min_value=1, max_value=5, step=1)
        center_tol = st.number_input("结晶峰中心拟合容差 (±°)",
                                       value=float(cryst_cfg.get("center_tolerance", 0.5)),
                                       min_value=0.1, max_value=2.0, step=0.05)

    if tt_max <= tt_min:
        st.error("2θ 上限必须大于下限")
        return

    # 裁切数据
    try:
        tt, inten = crop_range(tt_full, inten_full, tt_min, tt_max)
    except ValueError as e:
        st.error(str(e))
        return

    # ============ 3. 寻峰与峰表编辑 ============
    st.subheader("3️⃣ 峰列表")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        prominence_rel = st.slider("寻峰灵敏度 (相对最大强度)",
                                    min_value=0.005, max_value=0.20,
                                    value=float(fit_defaults.get("prominence", 0.02)),
                                    step=0.005,
                                    help="越小越敏感，会找到更多峰")
    with col_b:
        min_dist = st.slider("最小峰间距 (°)",
                              min_value=0.1, max_value=2.0,
                              value=float(fit_defaults.get("min_peak_distance", 0.3)),
                              step=0.05)
    with col_c:
        st.write("")
        st.write("")
        if st.button("🔍 自动寻峰（覆盖当前峰表）", use_container_width=True):
            try:
                peaks = auto_find_with_hints(
                    tt, inten,
                    config_path="config/materials.yaml",
                    prominence_rel=prominence_rel,
                    min_distance_deg=min_dist,
                    fwhm_min=float(cryst_cfg.get("fwhm_min", 0.1)),
                    fwhm_max=float(cryst_cfg.get("fwhm_max", 1.5)),
                )
                df_peaks = _peaks_to_df(peaks, default_kind="crystalline")

                # 自动追加 1 个无定形峰（如配置要求）
                n_amorph = int(amorph_cfg.get("count", 1))
                for _ in range(n_amorph):
                    df_peaks.loc[len(df_peaks)] = {
                        "use": True,
                        "kind": "amorphous",
                        "center": float(amorph_cfg.get("center_default", 21.0)),
                        "fwhm_est": 5.0,
                        "height": float(np.max(inten)) * 0.2,
                        "hint": "amorphous halo",
                    }
                st.session_state.single_peaks_df = df_peaks
                st.session_state.single_fit_result = None
                st.session_state.single_xc_result = None
                st.success(f"找到 {len(peaks)} 个结晶峰候选 + {n_amorph} 个无定形峰")
            except Exception as e:
                st.error(f"寻峰失败：{type(e).__name__}: {e}")

    # 初次进入：若无峰表，给一个空表 + 1 无定形峰
    if st.session_state.single_peaks_df is None:
        empty_df = pd.DataFrame(columns=["use", "kind", "center", "fwhm_est", "height", "hint"])
        empty_df.loc[0] = {
            "use": True, "kind": "amorphous",
            "center": float(amorph_cfg.get("center_default", 21.0)),
            "fwhm_est": 5.0,
            "height": float(np.max(inten)) * 0.2,
            "hint": "amorphous halo (default)",
        }
        st.session_state.single_peaks_df = empty_df

    st.markdown("**编辑峰表**（勾选 `use` 决定是否参与拟合；`kind` 选 `crystalline` 或 `amorphous`；"
                 "可直接增删行；`hint` 仅为参考标注）")

    edited_df = st.data_editor(
        st.session_state.single_peaks_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "use": st.column_config.CheckboxColumn("使用", default=True),
            "kind": st.column_config.SelectboxColumn(
                "类型",
                options=["crystalline", "amorphous"],
                required=True,
            ),
            "center": st.column_config.NumberColumn("中心 2θ (°)", format="%.4f", step=0.01),
            "fwhm_est": st.column_config.NumberColumn("FWHM 初值 (°)", format="%.3f", step=0.05),
            "height": st.column_config.NumberColumn("峰高初值", format="%.2f"),
            "hint": st.column_config.TextColumn("归属提示", help="仅参考，不影响拟合"),
        },
        key="single_peaks_editor",
    )
    st.session_state.single_peaks_df = edited_df

    n_cryst = int((edited_df["kind"] == "crystalline").sum()) if len(edited_df) else 0
    n_amorph = int((edited_df["kind"] == "amorphous").sum()) if len(edited_df) else 0
    st.caption(f"当前峰表：{n_cryst} 个结晶峰 + {n_amorph} 个无定形峰")

    # ============ 4. 执行拟合 ============
    st.subheader("4️⃣ 执行拟合")

    fix_centers = st.checkbox("固定所有结晶峰中心（高级；通常不勾选）",
                                value=False,
                                help="勾选后结晶峰中心不参与拟合，只优化宽度和幅度")

    if st.button("🚀 开始拟合", type="primary", use_container_width=True):
        cryst_peaks, amorph_peaks = _df_to_peak_lists(edited_df)
        if len(cryst_peaks) + len(amorph_peaks) == 0:
            st.error("峰表为空（或全部未勾选 use），无法拟合")
            return

        with st.spinner("拟合中..."):
            try:
                fit_res = build_and_fit(
                    tt, inten,
                    crystalline_peaks=cryst_peaks,
                    amorphous_peaks=amorph_peaks,
                    peak_shape=peak_shape,
                    baseline_kind=baseline_kind,
                    baseline_degree=int(baseline_degree),
                    center_tolerance=float(center_tol),
                    cryst_fwhm_range=(float(cryst_cfg.get("fwhm_min", 0.1)),
                                       float(cryst_cfg.get("fwhm_max", 1.5))),
                    amorph_fwhm_range=(float(amorph_cfg.get("fwhm_min", 1.5)),
                                        float(amorph_cfg.get("fwhm_max", 15.0))),
                    fix_centers=fix_centers,
                )
                xc_res = compute_crystallinity(fit_res)
                st.session_state.single_fit_result = fit_res
                st.session_state.single_xc_result = xc_res
                st.session_state.single_fit_settings = {
                    "peak_shape": peak_shape,
                    "baseline_kind": baseline_kind,
                    "baseline_degree": int(baseline_degree),
                    "tt_min": tt_min,
                    "tt_max": tt_max,
                    "center_tolerance": float(center_tol),
                    "fix_centers": fix_centers,
                }
                if fit_res["success"]:
                    st.success(f"✅ 拟合成功 | R² = {fit_res['r_squared']:.4f} | "
                               f"Xc = {xc_res['Xc_percent']:.2f}%")
                else:
                    st.warning(f"⚠️ 拟合可能未完全收敛：{fit_res['message']}")
            except Exception as e:
                st.error(f"拟合异常：{type(e).__name__}: {e}")
                return

    # ============ 5. 结果展示 ============
    fit_res = st.session_state.single_fit_result
    xc_res = st.session_state.single_xc_result

    if fit_res is None:
        st.info("配置好峰表后点击「开始拟合」查看结果。")
        return

    st.subheader("5️⃣ 拟合结果")

    # 指标
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("结晶度 Xc",
              f"{xc_res['Xc_percent']:.2f}%" if not np.isnan(xc_res['Xc_percent']) else "N/A")
    m2.metric("R²",
              f"{fit_res['r_squared']:.4f}" if not np.isnan(fit_res['r_squared']) else "N/A")
    m3.metric("Reduced χ²",
              f"{fit_res['reduced_chisqr']:.4g}" if not np.isnan(fit_res['reduced_chisqr']) else "N/A")
    m4.metric("结晶峰 / 无定形峰",
              f"{xc_res['n_crystalline_peaks']} / {xc_res['n_amorphous_peaks']}")

    # 图
    fig = plot_fit_result(
        fit_res, xc_res,
        title=meta["filename"],
        show_components=True,
    )
    st.pyplot(fig, use_container_width=True)

    # 峰参数表
    st.markdown("**拟合峰参数**")
    pk_rows = []
    for pk in fit_res["peak_info"]:
        pk_rows.append({
            "kind": pk["kind"],
            "init_center": round(pk["init_center"], 4),
            "fitted_center": round(pk["center"], 4) if not np.isnan(pk["center"]) else np.nan,
            "FWHM": round(pk["fwhm"], 4) if not np.isnan(pk["fwhm"]) else np.nan,
            "height": round(pk["height"], 3) if not np.isnan(pk["height"]) else np.nan,
            "area (amplitude)": round(pk["amplitude"], 3) if not np.isnan(pk["amplitude"]) else np.nan,
        })
    pk_df = pd.DataFrame(pk_rows)
    st.dataframe(pk_df, use_container_width=True)

    # Xc 详情
    with st.expander("📊 Xc 计算详情", expanded=False):
        st.json({
            "Xc": float(xc_res["Xc"]) if not np.isnan(xc_res["Xc"]) else None,
            "Xc_percent": float(xc_res["Xc_percent"]) if not np.isnan(xc_res["Xc_percent"]) else None,
            "area_crystalline": float(xc_res["area_crystalline"]),
            "area_amorphous": float(xc_res["area_amorphous"]),
            "area_total": float(xc_res["area_total"]),
            "method": xc_res["method"],
        })

    # lmfit 完整报告
    with st.expander("🔬 lmfit 完整拟合报告", expanded=False):
        if fit_res["lmfit_result"] is not None:
            st.code(fit_res["lmfit_result"].fit_report(), language="text")

    # ============ 6. 下载 ============
    st.subheader("6️⃣ 下载结果")

    base_name = meta["filename"].rsplit(".", 1)[0]

    dcol1, dcol2, dcol3 = st.columns(3)

    # 6a. 图 PNG
    with dcol1:
        png_bytes = fig_to_png_bytes(fig, dpi=200)
        st.download_button("📷 下载拟合图 (PNG)",
                            data=png_bytes,
                            file_name=f"{base_name}_fit.png",
                            mime="image/png",
                            use_container_width=True)

    # 6b. 峰参数 CSV
    with dcol2:
        csv_buf = io.StringIO()
        pk_df.to_csv(csv_buf, index=False)
        st.download_button("📊 下载峰参数 (CSV)",
                            data=csv_buf.getvalue(),
                            file_name=f"{base_name}_peaks.csv",
                            mime="text/csv",
                            use_container_width=True)

    # 6c. 完整 Excel（含原始数据、拟合曲线、分量、参数、Xc）
    with dcol3:
        excel_bytes = _build_single_excel(fit_res, xc_res, meta,
                                            st.session_state.single_fit_settings, pk_df)
        st.download_button("📦 下载完整结果 (Excel)",
                            data=excel_bytes,
                            file_name=f"{base_name}_result.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)


# ============================================================
# 构建单样品 Excel
# ============================================================
def _build_single_excel(fit_res, xc_res, meta, settings, pk_df) -> bytes:
    """打包单样品的所有结果到一个 Excel。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Summary
        summary = {
            "filename": meta["filename"],
            "n_points": meta["n_points"],
            "2theta_min_fit": settings["tt_min"] if settings else None,
            "2theta_max_fit": settings["tt_max"] if settings else None,
            "peak_shape": settings["peak_shape"] if settings else None,
            "baseline": settings["baseline_kind"] if settings else None,
            "Xc": xc_res["Xc"],
            "Xc_percent": xc_res["Xc_percent"],
            "area_crystalline": xc_res["area_crystalline"],
            "area_amorphous": xc_res["area_amorphous"],
            "n_crystalline_peaks": xc_res["n_crystalline_peaks"],
            "n_amorphous_peaks": xc_res["n_amorphous_peaks"],
            "R_squared": fit_res["r_squared"],
            "reduced_chisqr": fit_res["reduced_chisqr"],
            "fit_success": fit_res["success"],
            "fit_message": fit_res["message"],
        }
        pd.DataFrame([summary]).T.reset_index().rename(
            columns={"index": "field", 0: "value"}
        ).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Peaks
        pk_df.to_excel(writer, sheet_name="Peaks", index=False)

        # Sheet 3: Curves (原始 + 拟合 + 各分量)
        curves = pd.DataFrame({
            "two_theta": fit_res["two_theta"],
            "intensity_obs": fit_res["intensity"],
            "best_fit": fit_res["best_fit"],
            "residual": fit_res["intensity"] - fit_res["best_fit"],
            "crystalline_total": fit_res["crystalline_total"],
            "amorphous_total": fit_res["amorphous_total"],
        })
        if fit_res["baseline"] is not None:
            curves["baseline"] = fit_res["baseline"]
        # 各单峰分量
        for pk in fit_res["peak_info"]:
            prefix = pk["prefix"]
            comp = fit_res["components"].get(prefix)
            if comp is not None:
                col_name = f"{pk['kind'][:1]}_{prefix.rstrip('_')}_center{pk['center']:.2f}"
                curves[col_name] = comp
        curves.to_excel(writer, sheet_name="Curves", index=False)

        # Sheet 4: lmfit report
        report_text = fit_res["lmfit_result"].fit_report() if fit_res["lmfit_result"] else "N/A"
        pd.DataFrame({"lmfit_report": report_text.splitlines()}).to_excel(
            writer, sheet_name="FitReport", index=False
        )

    buf.seek(0)
    return buf.read()
