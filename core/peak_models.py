"""
峰型函数定义与 FWHM 转换工具。
所有峰型基于 lmfit 内置模型，确保参数定义一致。

注意：
- lmfit 的 PseudoVoigtModel 用 'sigma' 参数（不是 FWHM），需要换算
- Gaussian: FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma
- Lorentzian: FWHM = 2 * sigma
- PseudoVoigt (lmfit 实现): FWHM = 2 * sigma （sigma 已被 lmfit 内部归一化）
- Voigt: FWHM 由 sigma 和 gamma 共同决定，lmfit 直接提供 'fwhm' 派生参数
"""

import numpy as np
from lmfit.models import (
    GaussianModel,
    LorentzianModel,
    PseudoVoigtModel,
    VoigtModel,
    LinearModel,
    PolynomialModel,
)

# Gaussian 的 sigma -> FWHM 系数
GAUSSIAN_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.3548


PEAK_MODEL_MAP = {
    "Gaussian": GaussianModel,
    "Lorentzian": LorentzianModel,
    "PseudoVoigt": PseudoVoigtModel,
    "Voigt": VoigtModel,
}


def get_peak_model(shape: str, prefix: str):
    """
    根据峰型名称和前缀创建 lmfit 模型实例。

    Parameters
    ----------
    shape : str
        "Gaussian" / "Lorentzian" / "PseudoVoigt" / "Voigt"
    prefix : str
        参数前缀，例如 "p0_"、"amorph_"

    Returns
    -------
    lmfit.Model
    """
    if shape not in PEAK_MODEL_MAP:
        raise ValueError(f"Unsupported peak shape: {shape}. "
                         f"Choose from {list(PEAK_MODEL_MAP.keys())}")
    return PEAK_MODEL_MAP[shape](prefix=prefix)


def get_baseline_model(kind: str = "linear", degree: int = 1):
    """
    创建基线模型。

    Parameters
    ----------
    kind : str
        "linear" / "polynomial" / "none"
    degree : int
        多项式阶数（kind="polynomial" 时有效）

    Returns
    -------
    lmfit.Model or None
    """
    if kind == "none":
        return None
    if kind == "linear":
        return LinearModel(prefix="bkg_")
    if kind == "polynomial":
        return PolynomialModel(degree=degree, prefix="bkg_")
    raise ValueError(f"Unsupported baseline kind: {kind}")


def sigma_to_fwhm(sigma: float, shape: str) -> float:
    """sigma 转 FWHM（仅 Gaussian/Lorentzian/PseudoVoigt 用；Voigt 应直接读 fwhm 派生参数）。"""
    if shape == "Gaussian":
        return GAUSSIAN_FWHM_FACTOR * sigma
    if shape in ("Lorentzian", "PseudoVoigt"):
        return 2.0 * sigma
    raise ValueError(f"Use lmfit's derived 'fwhm' for shape={shape}")


def fwhm_to_sigma(fwhm: float, shape: str) -> float:
    """FWHM 转 sigma。"""
    if shape == "Gaussian":
        return fwhm / GAUSSIAN_FWHM_FACTOR
    if shape in ("Lorentzian", "PseudoVoigt"):
        return fwhm / 2.0
    raise ValueError(f"Cannot convert for shape={shape}; use lmfit param hints")


def get_peak_fwhm(result, prefix: str, shape: str) -> float:
    """
    从 lmfit 拟合结果中读取某峰的 FWHM。
    lmfit 对所有 4 种峰型都提供 'fwhm' 派生参数，优先使用。
    """
    fwhm_key = f"{prefix}fwhm"
    if fwhm_key in result.params:
        return float(result.params[fwhm_key].value)
    # fallback：从 sigma 计算
    sigma_key = f"{prefix}sigma"
    if sigma_key in result.params:
        return sigma_to_fwhm(float(result.params[sigma_key].value), shape)
    return np.nan


def get_peak_center(result, prefix: str) -> float:
    """从拟合结果中读取峰中心。"""
    key = f"{prefix}center"
    if key in result.params:
        return float(result.params[key].value)
    return np.nan


def get_peak_amplitude(result, prefix: str) -> float:
    """从拟合结果中读取峰幅度（积分面积）。lmfit 的 amplitude 即积分面积。"""
    key = f"{prefix}amplitude"
    if key in result.params:
        return float(result.params[key].value)
    return np.nan


def get_peak_height(result, prefix: str) -> float:
    """峰高（最大强度）。lmfit 提供 'height' 派生参数。"""
    key = f"{prefix}height"
    if key in result.params:
        return float(result.params[key].value)
    return np.nan
