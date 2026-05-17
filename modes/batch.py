"""
批量分析模式。
策略 A：用户配置一份统一峰表（"主峰表"），所有样品用同一套初值；
每个样品独立拟合，峰中心允许 ±center_tolerance 微调。
"""

import io
import os
import time
import numpy as np
import pandas as pd
import streamlit as st

from core.data_io import load_xrd_csv, crop_range
from core.peak_detection import auto_find_with_hints, load_materials_config
from core.fitting import build_and_fit
from core.crystallinity import compute_crystallinity
from core.plotting import plot_fit_result, fig_to_png_bytes, plot_batch_summary
from utils.export import build_summary_excel, build_batch_zip


# ============================================================
# Session State
# ============================================================
def _init_state():
    defaults = {
        "batch_files": [],              # list of {'name', 'bytes'}
        "batch_master_peaks_df": None,  # 主峰表 DataFrame
        "batch_results": None,          # list of result dicts
        "batch_failures": None,         # list of failure dicts
        "batch_settings": None,         # 拟合设置快照
        "batch_excel_bytes": None,
        "batch_zip_bytes": None,
        "batch_summary_fig_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# 主峰表辅助
# ============================================================
def _master_df_to_lists(df):
    """主峰表 DataFrame -> (crystalline_peaks, amorphous_peaks, peak_table_master)"""
    cryst, amorph, master = [], [], []
    if df is None or len(df) == 0:
        return cryst, amorph, master
    for i, row in df.iterrows():
        if not bool(row.get("use", True)):
            continue
        try:
            center = float(row["center"])
            fwhm = float(row["fwhm_est"])
            height = float(row.get("height_init", 100.0))
        except (ValueError, TypeError):
            continue
        kind = str(row.get("kind", "crystalline")).strip().lower()
        kind = "amorphous" if kind.startswith("amorph") else "crystalline"
        label = str(row.get("label", f"peak_{center:.2f}")).strip() or f"peak_{center:.2f}"

        entry = {"center": center, "fwhm_est": fwhm, "height": height}
        if kind == "amorphous":
            amorph.append(entry)
        else:
            cryst.append(entry)
        master.append({"label": label, "kind": kind, "center_init": center})
    return cryst, amorph, master


def _suggest_master_peaks_from_file(file_bytes, filename, cfg, tt_min, tt_max,
                                       prominence_rel, min_dist):
    """从一个参考样品自动寻峰，生成主峰表草稿。"""
    tt_full, inten_full, _ = load_xrd_csv(file_bytes, filename=filename)
    tt, inten = crop_range(tt_full, inten_full, tt_min, tt_max)

    cryst_cfg = cfg.get("crystalline_peak", {})
    amorph_cfg = cfg.get("amorphous_peak", {})

    peaks = auto_find_with_hints(
        tt, inten,
        config_path="config/materials.yaml",
        prominence_rel=prominence_rel,
        min_distance_deg=min_dist,
        fwhm_min=float(cryst_cfg.get("fwhm_min", 0.1)),
        fwhm_max=float(cryst_cfg.get("fwhm_max", 1.5)),
    )

    rows = []
    for i, p in enumerate(peaks):
        rows.append({
            "use": True,
            "label": f"C{i+1}_{p['center']:.2f}" + (f"_{p['hint']}" if p["hint"] and p["hint"] != "unknown" else ""),
            "kind": "crystalline",
            "center": round(p["center"], 3),
            "fwhm_est": round(p["fwhm_est"], 3),
            "height_init": round(p["height"], 2),
            "hint": p["hint"],
        })
    # 追加 1 个无定形峰
    n_amorph = int(amorph_cfg.get("count", 1))
    for k in range(n_amorph):
        rows.append({
            "use": True,
            "label": f"Amorph_{k+1}",
            "kind": "amorphous",
            "center": float(amorph_cfg.get("center_default", 21.0)),
            "fwhm_est": 5.0,
            "height_init": round(float(np.max(inten)) * 0.2, 2),
            "hint": "amorphous halo",
        })
    return pd.DataFrame(rows, columns=["use", "label", "kind", "center", "fwhm_est", "height_init", "hint"])


# ============================================================
# 单样品处理
# ============================================================
def _process_one_sample(file_bytes, filename, sample_index,
                         cryst_peaks, amorph_peaks,
                         settings, cryst_cfg, amorph_cfg):
    """处理单个样品，返回 (result_dict or None, failure_dict or None)"""
    try:
        tt_full, inten_full, meta = load_xrd_csv(file_bytes, filename=filename)
        tt, inten = crop_range(tt_full, inten_full,
                                 settings["tt_min"], settings["tt_max"])

        fit_res = build_and_fit(
            tt, inten,
            crystalline_peaks=cryst_peaks,
            amorphous_peaks=amorph_peaks,
            peak_shape=settings["peak_shape"],
            baseline_kind=settings["baseline_kind"],
            baseline_degree=settings["baseline_degree"],
            center_tolerance=settings["center_tolerance"],
            cryst_fwhm_range=(float(cryst_cfg.get("fwhm_min", 0.1)),
                               float(cryst_cfg.get("fwhm_max", 1.5))),
            amorph_fwhm_range=(float(amorph_cfg.get("fwhm_min", 1.5)),
                                float(amorph_cfg.get("fwhm_max", 15.0))),
            fix_centers=settings["fix_centers"],
        )

        if not fit_res["success"] and fit_res["lmfit_result"] is None:
            return None, {
                "sample_index": sample_index,
                "sample": _stem(filename),
                "error_type": "FitException",
                "error_msg": fit_res["message"],
            }

        xc_res = compute_crystallinity(fit_res)
        return {
            "sample_index": sample_index,
            "sample": _stem(filename),
            "meta": meta,
            "fit_result": fit_res,
            "xc_result": xc_res,
        }, None
    except Exception as e:
        return None, {
            "sample_index": sample_index,
            "sample": _stem(filename),
            "error_type": type(e).__name__,
            "error_msg": str(e),
        }


def _stem(filename: str) -> str:
    return os.path.splitext(os.path.basename(filename))[0]


# ============================================================
# 主渲染函数
# ============================================================
def render():
    _init_state()
    st.header("📚 批量分析模式")
    st.caption('策略：所有样品使用统一峰表（"主峰表"），每个样品独立拟合，'
           '...后面的内容...')

    cfg = load_materials_config("config/materials.yaml")
    fit_defaults = cfg.get("fitting_defaults", {})
    cryst_cfg = cfg.get("crystalline_peak", {})
    amorph_cfg = cfg.get("amorphous_peak", {})

    # ============ 1. 上传多文件 ============
    st.subheader("1️⃣ 上传所有样品 CSV")
    uploads = st.file_uploader("选择多个文件（可一次拖入多个）",
                                 type=["csv", "txt", "xy", "dat"],
                                 accept_multiple_files=True,
                                 key="batch_uploader")

    if uploads:
        # 缓存 bytes（防止 Streamlit 文件指针失效）
        st.session_state.batch_files = [
            {"name": f.name, "bytes": f.getvalue()} for f in uploads
        ]
        # 文件集合变了 → 清空旧结果
        st.session_state.batch_results = None
        st.session_state.batch_failures = None
        st.session_state.batch_excel_bytes = None
        st.session_state.batch_zip_bytes = None
        st.session_state.batch_summary_fig_bytes = None

    files = st.session_state.batch_files
    if not files:
        st.info("请上传至少 2 个 CSV 文件以使用批量模式（单个文件请用单文件模式）。")
        return

    st.success(f"已加载 {len(files)} 个文件")
    with st.expander("📋 文件列表", expanded=False):
        st.write([f["name"] for f in files])

    # ============ 2. 拟合设置 ============
    st.subheader("2️⃣ 全局拟合设置（应用于所有样品）")
    col1, col2, col3 = st.columns(3)
    with col1:
        tt_min = st.number_input("2θ 下限 (°)",
                                  value=float(fit_defaults.get("two_theta_min", 10.0)),
                                  step=0.5, key="batch_tt_min")
        tt_max = st.number_input("2θ 上限 (°)",
                                  value=float(fit_defaults.get("two_theta_max", 35.0)),
                                  step=0.5, key="batch_tt_max")
    with col2:
        peak_shape = st.selectbox("峰型",
                                   ["PseudoVoigt", "Gaussian", "Lorentzian", "Voigt"],
                                   index=["PseudoVoigt", "Gaussian", "Lorentzian", "Voigt"].index(
                                       fit_defaults.get("peak_shape", "PseudoVoigt")),
                                   key="batch_peak_shape")
        baseline_kind = st.selectbox("基线", ["linear", "polynomial", "none"],
                                       index=0, key="batch_baseline_kind")
    with col3:
        baseline_degree = st.number_input("多项式阶数",
                                            value=2, min_value=1, max_value=5, step=1,
                                            key="batch_baseline_degree")
        center_tol = st.number_input("结晶峰中心拟合容差 (±°)",
                                       value=float(cryst_cfg.get("center_tolerance", 0.5)),
                                       min_value=0.1, max_value=2.0, step=0.05,
                                       key="batch_center_tol")

    fix_centers = st.checkbox("固定结晶峰中心（不允许微调）",
                                value=False, key="batch_fix_centers",
                                help="勾选后所有样品峰中心严格使用主峰表的值")

    if tt_max <= tt_min:
        st.error("2θ 上限必须大于下限")
        return

    # ============ 3. 主峰表 ============
    st.subheader("3️⃣ 主峰表（应用于所有样品）")

    col_a, col_b = st.columns([2, 3])
    with col_a:
        ref_name = st.selectbox("从哪个样品自动生成主峰表草稿？",
                                  [f["name"] for f in files],
                                  key="batch_ref_sample")
        prom = st.slider("寻峰灵敏度",
                          0.005, 0.20,
                          float(fit_defaults.get("prominence", 0.02)), 0.005,
                          key="batch_prom")
        min_d = st.slider("最小峰间距 (°)",
                           0.1, 2.0,
                           float(fit_defaults.get("min_peak_distance", 0.3)), 0.05,
                           key="batch_min_d")
        if st.button("🔍 从所选样品自动生成主峰表", use_container_width=True):
            try:
                ref_file = next(f for f in files if f["name"] == ref_name)
                df_master = _suggest_master_peaks_from_file(
                    ref_file["bytes"], ref_file["name"], cfg,
                    tt_min, tt_max, prom, min_d
                )
                st.session_state.batch_master_peaks_df = df_master
                st.success(f"已生成主峰表（{len(df_master)} 个峰），请检查并编辑后再拟合")
            except Exception as e:
                st.error(f"生成失败：{type(e).__name__}: {e}")

    with col_b:
        st.markdown("**说明**：")
        st.markdown(
            "- 主峰表是所有样品共用的拟合初值。\n"
            "- `label` 用于在汇总报表中标识每个峰（建议填有意义的名字，如 `PTT_104`）。\n"
            "- `kind` 必须为 `crystalline` 或 `amorphous`。\n"
            "- 编辑完点最下方「开始批量拟合」。"
        )

    if st.session_state.batch_master_peaks_df is None:
        empty = pd.DataFrame(columns=["use", "label", "kind", "center",
                                         "fwhm_est", "height_init", "hint"])
        # 默认给 1 个无定形峰打底
        empty.loc[0] = {
            "use": True, "label": "Amorph_1", "kind": "amorphous",
            "center": float(amorph_cfg.get("center_default", 21.0)),
            "fwhm_est": 5.0, "height_init": 100.0,
            "hint": "amorphous halo (default)",
        }
        st.session_state.batch_master_peaks_df = empty

    edited_master = st.data_editor(
        st.session_state.batch_master_peaks_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "use": st.column_config.CheckboxColumn("使用", default=True),
            "label": st.column_config.TextColumn("标签", help="在汇总中用此名称"),
            "kind": st.column_config.SelectboxColumn(
                "类型", options=["crystalline", "amorphous"], required=True),
            "center": st.column_config.NumberColumn("中心 2θ (°)", format="%.4f", step=0.01),
            "fwhm_est": st.column_config.NumberColumn("FWHM 初值 (°)", format="%.3f", step=0.05),
            "height_init": st.column_config.NumberColumn("峰高初值", format="%.2f"),
            "hint": st.column_config.TextColumn("归属提示"),
        },
        key="batch_master_editor",
    )
    st.session_state.batch_master_peaks_df = edited_master

    cryst_peaks, amorph_peaks, master_table = _master_df_to_lists(edited_master)
    n_c = len(cryst_peaks)
    n_a = len(amorph_peaks)
    st.caption(f"主峰表当前：{n_c} 个结晶峰 + {n_a} 个无定形峰")

    # 标签唯一性检查
    labels = [m["label"] for m in master_table]
    if len(labels) != len(set(labels)):
        st.warning("⚠️ 主峰表中存在重复 label，汇总报表会出现列冲突。请修改为唯一名称。")

    # ============ 4. 执行批量拟合 ============
    st.subheader("4️⃣ 执行批量拟合")

    if st.button("🚀 开始批量拟合", type="primary", use_container_width=True):
        if n_c + n_a == 0:
            st.error("主峰表为空，无法拟合")
            return

        settings = {
            "tt_min": float(tt_min),
            "tt_max": float(tt_max),
            "peak_shape": peak_shape,
            "baseline_kind": baseline_kind,
            "baseline_degree": int(baseline_degree),
            "center_tolerance": float(center_tol),
            "fix_centers": bool(fix_centers),
        }
        st.session_state.batch_settings = settings

        results, failures = [], []
        progress = st.progress(0.0, text="初始化...")
        status_box = st.empty()

        t0 = time.time()
        total = len(files)
        for i, f in enumerate(files):
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (total - i) if i > 0 else 0
            status_box.info(f"[{i+1}/{total}] 处理中：{f['name']}  |  "
                              f"已用 {elapsed:.1f}s  |  预计剩余 {eta:.1f}s")
            ok, fail = _process_one_sample(
                f["bytes"], f["name"], i,
                cryst_peaks, amorph_peaks,
                settings, cryst_cfg, amorph_cfg,
            )
            if ok is not None:
                results.append(ok)
            if fail is not None:
                failures.append(fail)
            progress.progress((i + 1) / total,
                                text=f"完成 {i+1}/{total}")

        progress.empty()
        status_box.empty()

        st.session_state.batch_results = results
        st.session_state.batch_failures = failures

        # 构建汇总输出
        with st.spinner("生成汇总报表..."):
            excel_bytes = build_summary_excel(results, failures, master_table)
            st.session_state.batch_excel_bytes = excel_bytes

            # 跨样品汇总图
            try:
                summary_df = pd.DataFrame([
                    {
                        "sample": r["sample"],
                        "sample_index": r["sample_index"],
                        "Xc_percent": r["xc_result"]["Xc_percent"],
                        "R_squared": r["fit_result"]["r_squared"],
                    } for r in results
                ]).sort_values("sample_index")

                # 峰追踪：只画结晶峰（无定形峰位置参考价值低）
                tracking_rows = []
                for r in results:
                    for ref in master_table:
                        if ref["kind"] != "crystalline":
                            continue
                        from utils.export import _find_peak_by_init_center
                        best = _find_peak_by_init_center(
                            r["fit_result"]["peak_info"],
                            ref["center_init"], ref["kind"]
                        )
                        if best is None:
                            continue
                        tracking_rows.append({
                            "sample": r["sample"],
                            "sample_index": r["sample_index"],
                            "peak_label": ref["label"],
                            "center": best["center"],
                            "fwhm": best["fwhm"],
                        })
                tracking_df = pd.DataFrame(tracking_rows)

                if len(summary_df) > 0:
                    fig_summary = plot_batch_summary(summary_df, tracking_df)
                    summary_png = fig_to_png_bytes(fig_summary, dpi=180)
                    st.session_state.batch_summary_fig_bytes = summary_png
                else:
                    st.session_state.batch_summary_fig_bytes = None
            except Exception as e:
                st.warning(f"汇总图生成失败：{type(e).__name__}: {e}")
                st.session_state.batch_summary_fig_bytes = None

            # 每样品 PNG + CSV
            per_sample = []
            for r in results:
                try:
                    fig = plot_fit_result(r["fit_result"], r["xc_result"],
                                           title=r["sample"], show_components=True)
                    png = fig_to_png_bytes(fig, dpi=150)
                    import matplotlib.pyplot as plt
                    plt.close(fig)

                    # 峰参数 CSV
                    pk_rows = []
                    for pk in r["fit_result"]["peak_info"]:
                        pk_rows.append({
                            "kind": pk["kind"],
                            "init_center": pk["init_center"],
                            "fitted_center": pk["center"],
                            "fwhm": pk["fwhm"],
                            "height": pk["height"],
                            "area": pk["amplitude"],
                        })
                    csv_io = io.StringIO()
                    pd.DataFrame(pk_rows).to_csv(csv_io, index=False)

                    per_sample.append({
                        "sample": r["sample"],
                        "png_bytes": png,
                        "peaks_csv": csv_io.getvalue(),
                    })
                except Exception as e:
                    per_sample.append({
                        "sample": r["sample"],
                        "png_bytes": b"",
                        "peaks_csv": f"# Plot/CSV generation failed: {e}\n",
                    })

            zip_bytes = build_batch_zip(
                results, failures, master_table,
                excel_bytes, per_sample,
                batch_summary_png=st.session_state.batch_summary_fig_bytes,
            )
            st.session_state.batch_zip_bytes = zip_bytes

        st.success(f"✅ 批量拟合完成：成功 {len(results)} | 失败 {len(failures)} | "
                   f"耗时 {time.time()-t0:.1f}s")

    # ============ 5. 结果展示 ============
    results = st.session_state.batch_results
    failures = st.session_state.batch_failures
    if results is None:
        st.info("配置好主峰表后点击「开始批量拟合」。")
        return

    st.subheader("5️⃣ 批量结果")

    m1, m2, m3 = st.columns(3)
    m1.metric("成功样品数", len(results))
    m2.metric("失败样品数", len(failures))
    xc_arr = [r["xc_result"]["Xc_percent"] for r in results
                if not np.isnan(r["xc_result"]["Xc_percent"])]
    m3.metric("平均 Xc",
                f"{np.mean(xc_arr):.2f}%" if xc_arr else "N/A",
                f"±{np.std(xc_arr):.2f}%" if len(xc_arr) > 1 else None)

    # 汇总表
    if results:
        summary_view = pd.DataFrame([
            {
                "index": r["sample_index"],
                "sample": r["sample"],
                "Xc (%)": round(r["xc_result"]["Xc_percent"], 3),
                "R²": round(r["fit_result"]["r_squared"], 4),
                "χ²_red": round(r["fit_result"]["reduced_chisqr"], 4),
                "n_cryst": r["xc_result"]["n_crystalline_peaks"],
                "n_amorph": r["xc_result"]["n_amorphous_peaks"],
                "fit_ok": r["fit_result"]["success"],
            } for r in results
        ]).sort_values("index")
        st.markdown("**成功样品汇总**")
        st.dataframe(summary_view, use_container_width=True)

    if failures:
        st.markdown("**❌ 失败样品**")
        st.dataframe(pd.DataFrame(failures), use_container_width=True)

    # 汇总图
    if st.session_state.batch_summary_fig_bytes:
        st.markdown("**跨样品趋势图**")
        st.image(st.session_state.batch_summary_fig_bytes, use_container_width=True)

    # 单样品快速预览
    if results:
        with st.expander("🔍 查看单个样品拟合图", expanded=False):
            sample_names = [r["sample"] for r in results]
            sel = st.selectbox("选择样品", sample_names, key="batch_preview_select")
            r = next(rr for rr in results if rr["sample"] == sel)
            fig = plot_fit_result(r["fit_result"], r["xc_result"],
                                   title=r["sample"], show_components=True)
            st.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as plt
            plt.close(fig)

    # ============ 6. 下载 ============
    st.subheader("6️⃣ 下载结果")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        if st.session_state.batch_excel_bytes:
            st.download_button(
                "📊 下载汇总 Excel",
                data=st.session_state.batch_excel_bytes,
                file_name="batch_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with dcol2:
        if st.session_state.batch_zip_bytes:
            st.download_button(
                "📦 下载完整结果包 (ZIP)",
                data=st.session_state.batch_zip_bytes,
                file_name="batch_results.zip",
                mime="application/zip",
                use_container_width=True,
            )
