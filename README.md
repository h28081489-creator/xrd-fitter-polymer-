# XRD Crystallinity Analyzer for PTT/PBT Blends

专为 PTT / PBT 共混体系设计的 XRD 分峰与结晶度（Xc）分析工具。
基于 Streamlit + lmfit，支持单文件交互式分析与批量并行处理。

## 设计定位

本工具**不追求**通用高分子 XRD 分析能力，专注于：

- PTT/PBT 共混体系的**总结晶度**计算
- 自动寻峰 + 人工核验的高效工作流
- 批量样品的**峰位 / FWHM / 面积**跨样品趋势追踪（论文级输出）

## 功能特性

### 单文件模式
- CSV 数据加载（自动检测编码、分隔符、表头）
- 拟合范围可调（默认 10–35°）
- 自动寻峰 + 手动增删峰
- 4 种峰型函数：Pseudo-Voigt / Gaussian / Lorentzian / Voigt
- 基线扣除（线性 / 多项式）
- 拟合质量评估（R², χ², 残差图）
- Xc 计算与可视化

### 批量模式
- 多文件上传，统一峰表（策略 A：固定峰中心 ± 0.5° 微调）
- 串行处理，进度条实时反馈
- 失败样品容错（标记错误，不中断）
- 输出：
  - 汇总 Excel（4 个 Sheet：总览 / 峰参数矩阵 / 峰追踪 / 失败列表）
  - 每样品拟合 PNG
  - 跨样品趋势汇总图（Xc / R² / 峰位 / FWHM）
  - 全部打包为 zip 下载

## 安装

```bash
git clone <your-repo-url>
cd xrd_web
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
