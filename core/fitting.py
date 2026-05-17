"""
拟合主逻辑：构建多峰模型（结晶峰 + 无定形峰 + 基线），调用 lmfit 求解。
"""

import numpy as np
from lmfit import Parameters
from typing import List, Dict, Tuple, Optional

from .peak_models import (
    get_peak_model,
    get_baseline_model,
    get_peak_center,
    get_peak_fwhm,
    get_peak_amplitude,
    get_peak_height,
)


def build_and_fit(two_theta: np.ndarray,
                  intensity: np.ndarray,
                  crystalline_peaks: List[Dict],
                  amorphous_peaks: List[Dict],
                  peak_shape: str = "PseudoVoigt",
                  baseline_kind: str = "linear",
                  baseline_degree: int = 1,
                  center_tolerance: float = 0.5,
                  cryst_fwhm_range: Tuple[float, float] = (0.1, 1.5),
                  amorph_fwhm_range: Tuple[float, float] = (1.5, 15.0),
                  fix_centers: bool = False
                  ) -> Dict:
    """
    构建模型并拟合。

    Parameters
    ----------
    two_theta, intensity : np.ndarray
    crystalline_peaks : list of {'center': float, 'fwhm_est': float, 'height': float}
        结晶峰初值列表
    amorphous_peaks : list of {'center': float, 'fwhm_est': float, 'height': float}
        无定形峰初值列表（PTT/PBT 共混通常为 1 个）
    peak_shape : str
        "Gaussian" / "Lorentzian" / "PseudoVoigt" / "Voigt"
    baseline_kind : str
        "linear" / "polynomial" / "none"
    baseline_degree : int
        多项式基线阶数
    center_tolerance : float
        结晶峰中心的拟合微调范围（±度）
    cryst_fwhm_range, amorph_fwhm_range : (min, max)
        FWHM 边界
    fix_centers : bool
        若 True，固定所有结晶峰中心（批量模式策略 A 的严格版）

    Returns
    -------
    result_dict : dict
        {
            'lmfit_result': lmfit ModelResult,
            'two_theta': np.ndarray,
            'intensity': np.ndarray,
            'best_fit': np.ndarray,
            'components': dict[str, np.ndarray],  # 各分量 y 值
            'baseline': np.ndarray or None,
            'crystalline_total': np.ndarray,
            'amorphous_total': np.ndarray,
            'peak_info': list of dict,  # 每个峰的最终参数
            'r_squared': float,
            'reduced_chisqr': float,
            'success': bool,
            'message': str,
        }
    """
    if len(crystalline_peaks) + len(amorphous_peaks) == 0:
        raise ValueError("至少需要 1 个峰（结晶峰或无定形峰）")

    # === 1. 构建模型 ===
    composite = None
    params = Parameters()
    peak_prefixes = []  # [(prefix, kind, init_center), ...]

    # 1a. 基线
    bkg_model = get_baseline_model(baseline_kind, baseline_degree)
    if bkg_model is not None:
        composite = bkg_model
        # 基线初值：用数据边缘估计
        if baseline_kind == "linear":
            y_left = float(np.mean(intensity[:max(3, len(intensity)//20)]))
            y_right = float(np.mean(intensity[-max(3, len(intensity)//20):]))
            x_left = float(two_theta[0])
            x_right = float(two_theta[-1])
            slope_init = (y_right - y_left) / (x_right - x_left) if x_right > x_left else 0.0
            intercept_init = y_left - slope_init * x_left
            params.add("bkg_slope", value=slope_init)
            params.add("bkg_intercept", value=intercept_init, min=-abs(y_left)*10 - 1)
        elif baseline_kind == "polynomial":
            for i in range(baseline_degree + 1):
                params.add(f"bkg_c{i}", value=0.0)

    # 1b. 结晶峰
    for i, pk in enumerate(crystalline_peaks):
        prefix = f"c{i}_"
        model = get_peak_model(peak_shape, prefix)
        composite = model if composite is None else composite + model
        peak_prefixes.append((prefix, "crystalline", pk["center"]))

        center_init = float(pk["center"])
        fwhm_init = float(pk.get("fwhm_est", 0.3))
        fwhm_init = float(np.clip(fwhm_init, cryst_fwhm_range[0]*1.1, cryst_fwhm_range[1]*0.9))
        height_init = float(pk.get("height", float(np.max(intensity)) * 0.5))
        height_init = max(height_init, 1e-6)

        # sigma 初值（对所有受支持峰型，sigma ≈ FWHM/2，Gaussian 偏差小不影响收敛）
        sigma_init = fwhm_init / 2.0

        params.add(f"{prefix}center",
                   value=center_init,
                   min=center_init - center_tolerance,
                   max=center_init + center_tolerance,
                   vary=not fix_centers)
        params.add(f"{prefix}sigma",
                   value=sigma_init,
                   min=cryst_fwhm_range[0] / 2.5,
                   max=cryst_fwhm_range[1] / 1.5)
        # amplitude 初值：粗略 height * fwhm * 1.06 (Gaussian 系数)
        amp_init = height_init * fwhm_init * 1.06
        params.add(f"{prefix}amplitude",
                   value=amp_init,
                   min=0.0)

        # Voigt 额外需要 gamma
        if peak_shape == "Voigt":
            params.add(f"{prefix}gamma",
                       value=sigma_init,
                       min=cryst_fwhm_range[0] / 2.5,
                       max=cryst_fwhm_range[1] / 1.5,
                       vary=True)
        # PseudoVoigt 的 fraction（lmfit 默认 0.5，[0,1]）
        if peak_shape == "PseudoVoigt":
            params.add(f"{prefix}fraction", value=0.5, min=0.0, max=1.0)

    # 1c. 无定形峰
    for i, pk in enumerate(amorphous_peaks):
        prefix = f"a{i}_"
        model = get_peak_model(peak_shape, prefix)
        composite = model if composite is None else composite + model
        peak_prefixes.append((prefix, "amorphous", pk["center"]))

        center_init = float(pk["center"])
        fwhm_init = float(pk.get("fwhm_est", 5.0))
        fwhm_init = float(np.clip(fwhm_init, amorph_fwhm_range[0]*1.1, amorph_fwhm_range[1]*0.9))
        height_init = float(pk.get("height", float(np.max(intensity)) * 0.2))
        height_init = max(height_init, 1e-6)

        sigma_init = fwhm_init / 2.0

        # 无定形峰中心允许较大范围浮动
        params.add(f"{prefix}center",
                   value=center_init,
                   min=center_init - 3.0,
                   max=center_init + 3.0)
        params.add(f"{prefix}sigma",
                   value=sigma_init,
                   min=amorph_fwhm_range[0] / 2.5,
                   max=amorph_fwhm_range[1] / 1.5)
        amp_init = height_init * fwhm_init * 1.06
        params.add(f"{prefix}amplitude",
                   value=amp_init,
                   min=0.0)

        if peak_shape == "Voigt":
            params.add(f"{prefix}gamma",
                       value=sigma_init,
                       min=amorph_fwhm_range[0] / 2.5,
                       max=amorph_fwhm_range[1] / 1.5,
                       vary=True)
        if peak_shape == "PseudoVoigt":
            params.add(f"{prefix}fraction", value=0.5, min=0.0, max=1.0)

    # === 2. 拟合 ===
    try:
        result = composite.fit(intensity, params, x=two_theta,
                               method="leastsq", nan_policy="omit")
        success = result.success
        message = result.message or "OK"
    except Exception as e:
        return {
            "lmfit_result": None,
            "two_theta": two_theta,
            "intensity": intensity,
            "best_fit": np.full_like(intensity, np.nan),
            "components": {},
            "baseline": None,
            "crystalline_total": np.zeros_like(intensity),
            "amorphous_total": np.zeros_like(intensity),
            "peak_info": [],
            "r_squared": np.nan,
            "reduced_chisqr": np.nan,
            "success": False,
            "message": f"拟合异常：{type(e).__name__}: {e}",
        }

    # === 3. 提取分量 ===
    components = result.eval_components(x=two_theta)
    best_fit = result.best_fit

    baseline = components.get("bkg_", None)
    cryst_total = np.zeros_like(intensity, dtype=float)
    amorph_total = np.zeros_like(intensity, dtype=float)
    peak_info = []

    for prefix, kind, init_center in peak_prefixes:
        comp = components.get(prefix, None)
        if comp is None:
            continue
        if kind == "crystalline":
            cryst_total = cryst_total + comp
        else:
            amorph_total = amorph_total + comp

        peak_info.append({
            "prefix": prefix,
            "kind": kind,
            "init_center": init_center,
            "center": get_peak_center(result, prefix),
            "fwhm": get_peak_fwhm(result, prefix, peak_shape),
            "amplitude": get_peak_amplitude(result, prefix),  # 积分面积
            "height": get_peak_height(result, prefix),
        })

    # === 4. 拟合质量 ===
    residual = intensity - best_fit
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((intensity - np.mean(intensity)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    red_chi = float(result.redchi) if result.redchi is not None else np.nan

    return {
        "lmfit_result": result,
        "two_theta": two_theta,
        "intensity": intensity,
        "best_fit": best_fit,
        "components": components,
        "baseline": baseline,
        "crystalline_total": cryst_total,
        "amorphous_total": amorph_total,
        "peak_info": peak_info,
        "r_squared": r_squared,
        "reduced_chisqr": red_chi,
        "success": success,
        "message": message,
    }
