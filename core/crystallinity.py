"""
结晶度 Xc 计算。
PTT/PBT 共混：只计算总结晶度 Xc = A_crystalline / (A_crystalline + A_amorphous)
"""

import numpy as np
from typing import Dict


def compute_crystallinity(fit_result: Dict) -> Dict:
    """
    从 build_and_fit 的返回值计算 Xc。

    Parameters
    ----------
    fit_result : dict
        build_and_fit 返回的字典

    Returns
    -------
    xc_result : dict
        {
            'Xc': float,                    # 结晶度（0-1，乘 100 即百分比）
            'Xc_percent': float,            # 百分比形式
            'area_crystalline': float,      # 结晶峰总积分面积
            'area_amorphous': float,        # 无定形峰总积分面积
            'area_total': float,
            'n_crystalline_peaks': int,
            'n_amorphous_peaks': int,
            'method': str,
        }
    """
    peak_info = fit_result.get("peak_info", [])

    area_cryst = 0.0
    area_amorph = 0.0
    n_c = 0
    n_a = 0

    for pk in peak_info:
        amp = pk.get("amplitude", np.nan)
        if np.isnan(amp) or amp < 0:
            continue
        if pk["kind"] == "crystalline":
            area_cryst += amp
            n_c += 1
        elif pk["kind"] == "amorphous":
            area_amorph += amp
            n_a += 1

    area_total = area_cryst + area_amorph
    if area_total > 0:
        xc = area_cryst / area_total
    else:
        xc = np.nan

    return {
        "Xc": xc,
        "Xc_percent": xc * 100.0 if not np.isnan(xc) else np.nan,
        "area_crystalline": area_cryst,
        "area_amorphous": area_amorph,
        "area_total": area_total,
        "n_crystalline_peaks": n_c,
        "n_amorphous_peaks": n_a,
        "method": "Integrated peak areas (lmfit amplitude)",
    }
