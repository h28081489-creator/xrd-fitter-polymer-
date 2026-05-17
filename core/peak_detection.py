"""
自动寻峰 + PTT/PBT 峰位归属提示。
归属提示仅做信息标注，不影响拟合。
"""

import os
import yaml
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Optional


def load_materials_config(config_path: str = "config/materials.yaml") -> dict:
    """加载 PTT/PBT 峰位配置。"""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def auto_find_peaks(two_theta: np.ndarray,
                    intensity: np.ndarray,
                    prominence_rel: float = 0.02,
                    min_distance_deg: float = 0.3,
                    fwhm_min: float = 0.1,
                    fwhm_max: float = 1.5
                    ) -> List[Dict]:
    """
    自动寻峰。

    Parameters
    ----------
    two_theta, intensity : np.ndarray
    prominence_rel : float
        相对最大强度的 prominence 阈值（0.02 = 2% 最大强度）
    min_distance_deg : float
        最小峰间距（度），换算为采样点数
    fwhm_min, fwhm_max : float
        粗筛峰宽范围（基于 scipy 给出的 width 估计）

    Returns
    -------
    peaks : list of dict
        [{'center': float, 'height': float, 'fwhm_est': float, 'index': int}, ...]
    """
    if len(two_theta) < 5:
        return []

    intensity = np.asarray(intensity, dtype=float)
    max_int = float(np.max(intensity))
    if max_int <= 0:
        return []

    # 度 -> 采样点数
    step = float(np.median(np.diff(two_theta)))
    if step <= 0:
        return []
    distance_pts = max(1, int(round(min_distance_deg / step)))

    prom_abs = prominence_rel * max_int

    # scipy find_peaks
    indices, props = find_peaks(
        intensity,
        prominence=prom_abs,
        distance=distance_pts,
        width=1,  # 至少 1 个点宽
    )

    peaks = []
    widths = props.get("widths", np.array([]))
    for i, idx in enumerate(indices):
        center = float(two_theta[idx])
        height = float(intensity[idx])
        fwhm_est = float(widths[i] * step) if i < len(widths) else np.nan

        # FWHM 粗筛（仅在能估计时）
        if not np.isnan(fwhm_est):
            if fwhm_est < fwhm_min * 0.5 or fwhm_est > fwhm_max * 1.5:
                continue

        peaks.append({
            "center": center,
            "height": height,
            "fwhm_est": fwhm_est if not np.isnan(fwhm_est) else 0.3,
            "index": int(idx),
        })

    # 按强度降序排列
    peaks.sort(key=lambda p: p["height"], reverse=True)
    return peaks


def assign_peak_hints(peaks: List[Dict],
                      materials_cfg: dict
                      ) -> List[Dict]:
    """
    为每个检测到的峰附加 PTT/PBT 归属提示（仅信息，不修改拟合参数）。

    Returns
    -------
    peaks_with_hint : list of dict
        在原 dict 基础上加 'hint' 字段，例如 'PTT(012)' 或 'PBT(011) / PTT(-112)' 或 'unknown'
    """
    hint_cfg = materials_cfg.get("peak_assignment_hint", {})
    if not hint_cfg.get("enabled", True):
        for p in peaks:
            p["hint"] = ""
        return peaks

    tol = float(hint_cfg.get("tolerance", 0.3))
    ptt_list = materials_cfg.get("PTT_peaks", []) or []
    pbt_list = materials_cfg.get("PBT_peaks", []) or []

    for p in peaks:
        c = p["center"]
        matches = []
        for ref in ptt_list:
            if abs(c - float(ref["center"])) <= tol:
                hkl = ref.get("hkl", "")
                matches.append(f"PTT{hkl}")
        for ref in pbt_list:
            if abs(c - float(ref["center"])) <= tol:
                hkl = ref.get("hkl", "")
                matches.append(f"PBT{hkl}")
        p["hint"] = " / ".join(matches) if matches else "unknown"

    return peaks


def auto_find_with_hints(two_theta: np.ndarray,
                          intensity: np.ndarray,
                          config_path: str = "config/materials.yaml",
                          **kwargs) -> List[Dict]:
    """便捷函数：寻峰 + 归属提示，一步到位。"""
    cfg = load_materials_config(config_path)
    fit_defaults = cfg.get("fitting_defaults", {})
    cryst_cfg = cfg.get("crystalline_peak", {})

    params = {
        "prominence_rel": kwargs.get("prominence_rel",
                                      fit_defaults.get("prominence", 0.02)),
        "min_distance_deg": kwargs.get("min_distance_deg",
                                        fit_defaults.get("min_peak_distance", 0.3)),
        "fwhm_min": kwargs.get("fwhm_min",
                                cryst_cfg.get("fwhm_min", 0.1)),
        "fwhm_max": kwargs.get("fwhm_max",
                                cryst_cfg.get("fwhm_max", 1.5)),
    }
    peaks = auto_find_peaks(two_theta, intensity, **params)
    peaks = assign_peak_hints(peaks, cfg)
    return peaks
