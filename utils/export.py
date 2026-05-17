"""
批量结果导出工具：Excel 汇总 + ZIP 打包。
"""

import io
import zipfile
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def build_summary_excel(results: List[Dict],
                          failures: List[Dict],
                          peak_table_master: Optional[List[Dict]] = None
                          ) -> bytes:
    """
    构建批量汇总 Excel（4 个 Sheet）。

    Parameters
    ----------
    results : list of dict
        每个元素包含成功样品的：
        {
            'sample': str,             # 样品名（文件名 stem）
            'sample_index': int,
            'meta': dict,
            'fit_result': dict,        # build_and_fit 的返回值
            'xc_result': dict,         # compute_crystallinity 的返回值
        }
    failures : list of dict
        失败样品：[{'sample', 'sample_index', 'error_type', 'error_msg'}, ...]
    peak_table_master : list of dict or None
        批量模式使用的统一峰表（含 center / kind / label），用于"峰追踪"对齐。
        每行包含：{'label': str, 'kind': str, 'center_init': float}

    Returns
    -------
    bytes : Excel 文件二进制
    """
    # === Sheet 1: Summary ===
    summary_rows = []
    for r in results:
        fit = r["fit_result"]
        xc = r["xc_result"]
        summary_rows.append({
            "sample_index": r["sample_index"],
            "sample": r["sample"],
            "n_points": r["meta"]["n_points"],
            "Xc": xc["Xc"],
            "Xc_percent": xc["Xc_percent"],
            "area_crystalline": xc["area_crystalline"],
            "area_amorphous": xc["area_amorphous"],
            "n_crystalline_peaks": xc["n_crystalline_peaks"],
            "n_amorphous_peaks": xc["n_amorphous_peaks"],
            "R_squared": fit["r_squared"],
            "reduced_chisqr": fit["reduced_chisqr"],
            "fit_success": fit["success"],
            "fit_message": fit["message"],
        })
    # 把失败样品也加进 summary（Xc 留空）
    for f in failures:
        summary_rows.append({
            "sample_index": f["sample_index"],
            "sample": f["sample"],
            "n_points": np.nan,
            "Xc": np.nan,
            "Xc_percent": np.nan,
            "area_crystalline": np.nan,
            "area_amorphous": np.nan,
            "n_crystalline_peaks": 0,
            "n_amorphous_peaks": 0,
            "R_squared": np.nan,
            "reduced_chisqr": np.nan,
            "fit_success": False,
            "fit_message": f"FAILED: {f['error_type']}: {f['error_msg']}",
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("sample_index")

    # === Sheet 2: Peak Parameters（宽表）===
    # 每行一个样品，列为每个峰的 center/fwhm/area/height
    peak_param_rows = []
    for r in results:
        row = {"sample_index": r["sample_index"], "sample": r["sample"]}
        peak_info = r["fit_result"]["peak_info"]
        # 用 peak_table_master 的 label 对齐
        if peak_table_master is not None:
            # 按初始 center 匹配（最近邻）
            for ref in peak_table_master:
                ref_label = ref["label"]
                ref_center = ref["center_init"]
                ref_kind = ref["kind"]
                best = _find_peak_by_init_center(peak_info, ref_center, ref_kind)
                if best is not None:
                    row[f"{ref_label}_center"] = best["center"]
                    row[f"{ref_label}_fwhm"] = best["fwhm"]
                    row[f"{ref_label}_area"] = best["amplitude"]
                    row[f"{ref_label}_height"] = best["height"]
                else:
                    row[f"{ref_label}_center"] = np.nan
                    row[f"{ref_label}_fwhm"] = np.nan
                    row[f"{ref_label}_area"] = np.nan
                    row[f"{ref_label}_height"] = np.nan
        else:
            for i, pk in enumerate(peak_info):
                base = f"peak{i+1}_{pk['kind'][:1]}"
                row[f"{base}_center"] = pk["center"]
                row[f"{base}_fwhm"] = pk["fwhm"]
                row[f"{base}_area"] = pk["amplitude"]
                row[f"{base}_height"] = pk["height"]
        peak_param_rows.append(row)
    peak_param_df = pd.DataFrame(peak_param_rows).sort_values("sample_index")

    # === Sheet 3: Peak Tracking（长表，便于趋势绘图）===
    tracking_rows = []
    for r in results:
        peak_info = r["fit_result"]["peak_info"]
        if peak_table_master is not None:
            for ref in peak_table_master:
                best = _find_peak_by_init_center(peak_info, ref["center_init"], ref["kind"])
                if best is None:
                    continue
                tracking_rows.append({
                    "sample_index": r["sample_index"],
                    "sample": r["sample"],
                    "peak_label": ref["label"],
                    "kind": ref["kind"],
                    "init_center": ref["center_init"],
                    "center": best["center"],
                    "fwhm": best["fwhm"],
                    "area": best["amplitude"],
                    "height": best["height"],
                })
        else:
            for i, pk in enumerate(peak_info):
                tracking_rows.append({
                    "sample_index": r["sample_index"],
                    "sample": r["sample"],
                    "peak_label": f"peak{i+1}",
                    "kind": pk["kind"],
                    "init_center": pk["init_center"],
                    "center": pk["center"],
                    "fwhm": pk["fwhm"],
                    "area": pk["amplitude"],
                    "height": pk["height"],
                })
    tracking_df = pd.DataFrame(tracking_rows)

    # === Sheet 4: Failures ===
    failures_df = pd.DataFrame(failures) if failures else pd.DataFrame(
        columns=["sample_index", "sample", "error_type", "error_msg"]
    )

    # === 写入 Excel ===
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        peak_param_df.to_excel(writer, sheet_name="PeakParameters", index=False)
        tracking_df.to_excel(writer, sheet_name="PeakTracking", index=False)
        failures_df.to_excel(writer, sheet_name="Failures", index=False)

    buf.seek(0)
    return buf.read()


def _find_peak_by_init_center(peak_info: List[Dict],
                                 target_center: float,
                                 target_kind: str,
                                 tol: float = 1.0) -> Optional[Dict]:
    """在拟合结果的 peak_info 中找与目标初始 center 最匹配的峰（同类型优先）。"""
    same_kind = [p for p in peak_info if p["kind"] == target_kind]
    candidates = same_kind if same_kind else peak_info
    best = None
    best_diff = float("inf")
    for p in candidates:
        # 优先用 init_center 匹配（批量模式下 init_center 直接对应主峰表）
        diff = abs(p["init_center"] - target_center)
        if diff < best_diff and diff <= tol:
            best = p
            best_diff = diff
    return best


def build_batch_zip(results: List[Dict],
                     failures: List[Dict],
                     peak_table_master: Optional[List[Dict]],
                     summary_excel_bytes: bytes,
                     per_sample_pngs: List[Dict],
                     batch_summary_png: Optional[bytes] = None
                     ) -> bytes:
    """
    打包批量输出：
      - summary.xlsx
      - per_sample/{sample}_fit.png  (每样品拟合图)
      - batch_summary.png            (4 子图汇总)
      - per_sample/{sample}_peaks.csv (每样品峰参数)

    Parameters
    ----------
    per_sample_pngs : list of dict
        [{'sample': str, 'png_bytes': bytes, 'peaks_csv': str}, ...]
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.xlsx", summary_excel_bytes)
        if batch_summary_png is not None:
            zf.writestr("batch_summary.png", batch_summary_png)
        for item in per_sample_pngs:
            stem = item["sample"]
            zf.writestr(f"per_sample/{stem}_fit.png", item["png_bytes"])
            if "peaks_csv" in item and item["peaks_csv"]:
                zf.writestr(f"per_sample/{stem}_peaks.csv", item["peaks_csv"])

        # README
        readme = (
            "Batch XRD Fitting Results\n"
            "=========================\n\n"
            "Files:\n"
            "  summary.xlsx          - 4 sheets: Summary / PeakParameters / PeakTracking / Failures\n"
            "  batch_summary.png     - Cross-sample trend plots (Xc, R^2, peak center, FWHM)\n"
            "  per_sample/*_fit.png  - Individual fit visualization per sample\n"
            "  per_sample/*_peaks.csv- Individual peak parameter table per sample\n\n"
            f"Total samples: {len(results) + len(failures)}\n"
            f"Successful  : {len(results)}\n"
            f"Failed      : {len(failures)}\n"
        )
        zf.writestr("README.txt", readme)

    buf.seek(0)
    return buf.read()
