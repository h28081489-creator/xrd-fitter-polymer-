"""
CSV 数据加载：自动检测编码、分隔符、表头。
"""

import io
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def _try_decode(raw: bytes) -> Tuple[str, str]:
    """尝试多种编码解码，返回 (text, encoding_used)。"""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # 最后兜底
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _detect_separator(sample_text: str) -> str:
    """从样本文本中检测分隔符。"""
    # 取前 20 行非空、非注释行做统计
    lines = [ln for ln in sample_text.splitlines()
             if ln.strip() and not ln.strip().startswith(("#", "%", "//", ";"))]
    sample = "\n".join(lines[:20])

    candidates = [",", "\t", ";", r"\s+"]
    best_sep = ","
    best_count = 0
    for sep in candidates:
        try:
            df = pd.read_csv(io.StringIO(sample), sep=sep, header=None,
                             engine="python", on_bad_lines="skip", nrows=10)
            if df.shape[1] >= 2:
                # 优先选择列数稳定为 2 的分隔符
                if df.shape[1] == 2 and df.shape[0] > best_count:
                    best_sep = sep
                    best_count = df.shape[0]
                elif best_count == 0 and df.shape[1] >= 2:
                    best_sep = sep
        except Exception:
            continue
    return best_sep


def _has_header(sample_text: str, sep: str) -> bool:
    """判断第一行是否为表头（首行不可全部转为浮点则视为表头）。"""
    lines = [ln for ln in sample_text.splitlines()
             if ln.strip() and not ln.strip().startswith(("#", "%", "//", ";"))]
    if not lines:
        return False
    first = lines[0]
    if sep == r"\s+":
        tokens = first.split()
    else:
        tokens = first.split(sep)
    if len(tokens) < 2:
        return False
    try:
        float(tokens[0].strip())
        float(tokens[1].strip())
        return False  # 首行可解析为数字 -> 无表头
    except ValueError:
        return True


def load_xrd_csv(file_input, filename: str = "uploaded.csv"
                 ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    加载 XRD CSV 数据。

    Parameters
    ----------
    file_input : bytes / str(path) / file-like
        Streamlit uploaded file 的 .getvalue() 返回 bytes；也支持本地路径或 BytesIO
    filename : str
        用于报告的文件名

    Returns
    -------
    two_theta : np.ndarray  (1D, 升序)
    intensity : np.ndarray  (1D)
    meta : dict
        包含 encoding / separator / has_header / n_points / filename
    """
    # 1. 读为 bytes
    if isinstance(file_input, (bytes, bytearray)):
        raw = bytes(file_input)
    elif hasattr(file_input, "read"):
        raw = file_input.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    elif isinstance(file_input, str):
        with open(file_input, "rb") as f:
            raw = f.read()
    else:
        raise TypeError(f"Unsupported file_input type: {type(file_input)}")

    # 2. 解码
    text, encoding = _try_decode(raw)

    # 3. 检测分隔符与表头
    sep = _detect_separator(text)
    has_hdr = _has_header(text, sep)

    # 4. 读取
    read_kwargs = dict(
        sep=sep,
        engine="python",
        comment="#",
        skip_blank_lines=True,
        on_bad_lines="skip",
    )
    if has_hdr:
        df = pd.read_csv(io.StringIO(text), header=0, **read_kwargs)
    else:
        df = pd.read_csv(io.StringIO(text), header=None, **read_kwargs)

    if df.shape[1] < 2:
        raise ValueError(
            f"无法解析为至少 2 列数据（检测到 {df.shape[1]} 列）。"
            f"请检查文件格式：encoding={encoding}, sep={repr(sep)}"
        )

    # 5. 取前两列，强制转 float
    df = df.iloc[:, :2].copy()
    df.columns = ["two_theta", "intensity"]
    df["two_theta"] = pd.to_numeric(df["two_theta"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.dropna()

    if len(df) < 10:
        raise ValueError(f"有效数据点过少 ({len(df)} 行)，无法拟合")

    # 6. 排序 + 去重
    df = df.sort_values("two_theta").drop_duplicates(subset="two_theta")

    two_theta = df["two_theta"].to_numpy(dtype=float)
    intensity = df["intensity"].to_numpy(dtype=float)

    meta = {
        "filename": filename,
        "encoding": encoding,
        "separator": sep,
        "has_header": has_hdr,
        "n_points": len(two_theta),
        "two_theta_min": float(two_theta.min()),
        "two_theta_max": float(two_theta.max()),
    }
    return two_theta, intensity, meta


def crop_range(two_theta: np.ndarray, intensity: np.ndarray,
               tt_min: float, tt_max: float
               ) -> Tuple[np.ndarray, np.ndarray]:
    """裁切到指定 2θ 范围。"""
    mask = (two_theta >= tt_min) & (two_theta <= tt_max)
    if mask.sum() < 10:
        raise ValueError(
            f"裁切后数据点过少 ({mask.sum()} 点)。"
            f"数据范围 [{two_theta.min():.2f}, {two_theta.max():.2f}]，"
            f"请求范围 [{tt_min}, {tt_max}]"
        )
    return two_theta[mask], intensity[mask]
