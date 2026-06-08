# NR TDD 语义通信物理层仿真系统

基于 5G NR TDD 帧结构的端到端语义通信仿真平台，实现基站(BS)到用户设备(UE)的 SwinJSCC 语义图像传输与 CSI 反馈闭环。

## 项目简介

本项目模拟了 5G NR TDD 模式下物理层的关键流程：

1. **下行链路（DL）**：基站使用 SwinJSCC 深度语义编码器压缩图像，经 OFDM 调制后通过无线信道发送。同时传输传统 H.264+LDPC 编码作为对比基线。
2. **上行链路（UL）**：UE 接收后进行信道估计（LS 估计 + 频域/时域插值），将 CSI 压缩后通过上行时隙反馈给基站。
3. **信道模型**：支持 AWGN、瑞利多径(Rayleigh TDL)、莱斯多径(Rician TDL)三种信道。
4. **性能评估**：输出语义/传统方案的 PSNR、星座图对比、时频资源网格可视化等。

**应用场景**：面向 6G 语义通信研究，提供可配置的物理层仿真环境，支持算法对比与验证。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     NR TDD 帧结构                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──┐ ┌──┐      │
│  │ DL Slot 0│  │ DL Slot 1│  │ DL Slot..│..│UL│ │UL│      │
│  │ 14 sym   │  │ 14 sym   │  │          │  │S0│ │S1│      │
│  └──────────┘  └──────────┘  └──────────┘  └──┘ └──┘      │
│   ◄──────── 72 DL slots ────────►  ◄─ 2 UL slots ─►       │
│                                                             │
│  每符号：子载波上插入梳状导频 (1/4 RE)，用于信道估计         │
└─────────────────────────────────────────────────────────────┘

┌──────────┐                              ┌──────────┐
│    BS    │──── DL: SwinJSCC + H.264 ───►│    UE    │
│ (基站)   │◄── UL: CSI 压缩反馈 ────────│ (用户)   │
└──────────┘                              └──────────┘
```

### 数据流

```
图像 → SwinJSCC编码 → QPSK调制 → 资源网格映射 → OFDM调制
  │                                                       │
  │  ┌──────────────── AWGN/Rayleigh/Rician 信道 ────────┐ │
  ▼                                                       ▼
图像 ← SwinJSCC解码 ← QPSK解调 ← 信道均衡 ← OFDM解调 ←────┘

同时：
原始图像 → H.264编码 → LDPC编码 → QPSK → 网格映射 → OFDM → 信道
  → OFDM解调 → QPSK解调 → LDPC解码 → H.264解码 → 传统重建图像

UL CSI反馈：
UE信道估计(H_ls) → CSI压缩(降采样+QPSK) → UL时隙传输
  → BS检测恢复 → CSI重建(NMSE评估)
```

## 目录结构

```
nr_tdd_semantic/
├── __init__.py          # 包初始化
├── config.py            # 全局配置数据类
├── channel.py           # 多径信道模型 (AWGN/Rayleigh/Rician)
├── conventional.py      # H.264 + LDPC 传统编解码方案
├── dsp.py               # 信道估计 (LS)、CSI 压缩/解压
├── learned_estimator.py # 复值通信基座模型推理包装
├── models/              # 复值层、通信 backbone、任务头
├── nodes.py             # BS 基站与 UE 用户设备节点
├── ofdm.py              # OFDM 调制/解调
├── resource_grid.py     # 时频资源网格映射 (导频+数据)
├── semantic.py          # SwinJSCC 语义编解码接口
├── simulation.py        # 端到端仿真主循环
├── scripts/             # 数据集导出与模型训练脚本
├── reports/             # 工程探索报告
├── utils.py             # 工具函数 (QPSK调制解调、复数数组等)
├── visualization.py     # 可视化 (星座图、PSNR对比、资源网格)
├── run_demo.py          # 命令行入口 🚀
└── outputs/             # 仿真输出目录 (自动生成)
```

## 快速开始

### 环境要求

- Python ≥ 3.9
- NumPy, SciPy
- PyTorch (用于 SwinJSCC 模型)
- Pillow
- Matplotlib (可视化)
- PyAV (H.264 编解码，可选)
- SwinJSCC 模型文件 (放在 `/root/lap/semantic/SwinJSCC/`)

### 安装依赖

```bash
pip install numpy scipy torch pillow matplotlib av
```

### 运行仿真

```bash
# 进入项目目录
cd nr_tdd_semantic

# AWGN 信道, SNR=20dB
python run_demo.py --channel awgn --snr-db 20

# 瑞利衰落信道
python run_demo.py --channel rayleigh --snr-db 15

# 莱斯衰落信道
python run_demo.py --channel rician --snr-db 25 --rician-k-db 9

# 完整参数示例
python run_demo.py \
    --image /path/to/image.png \
    --channel rayleigh \
    --snr-db 18 \
    --scs-khz 30 \
    --n-fft 2048 \
    --n-subcarriers 1200 \
    --dl-slots 72 \
    --ul-slots 2 \
    --delays "0,3,7" \
    --powers-db "0,-3,-9" \
    --seed 42 \
    --semantic swinjscc \
    --h264-crf 28 \
    --channel-estimator ls

# 使用复值通信基座模型信道估计器
python run_demo.py \
    --channel rayleigh \
    --snr-db 20 \
    --channel-estimator comm_foundation \
    --comm-foundation-checkpoint outputs/comm_foundation_ckpt_residual_safe/best_comm_foundation_channel_estimator.pt
```

`comm_foundation` 路径采用保守增强策略：先用 LS 得到 `H_ls`，再根据通信结构做块静态 CSI 平滑，最后由零初始化 residual 复值网络输出修正量。弱 checkpoint 不会覆盖 LS 先验；训练充分时 residual head 可以继续学习更细的 CSI 修正。

### 通信基座模型数据集与训练

```bash
# 导出 CSI/IQ sanity 数据集
python scripts/export_comm_foundation_dataset.py \
    --num_samples 100 \
    --snr_min 0 \
    --snr_max 30 \
    --channel_mode mixed \
    --output_path outputs/comm_foundation_sanity_100.npz

# 训练最小复值 backbone + 信道估计 head
python scripts/train_comm_foundation_model.py \
    --dataset_path outputs/comm_foundation_sanity_100.npz \
    --output_dir outputs/comm_foundation_ckpt_residual_safe \
    --epochs 1 \
    --batch_size 16 \
    --use_masked_csi \
    --use_denoising
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--image` | str | Kodak kodim01 | 输入图像路径 |
| `--channel` | choice | rayleigh | 信道模型：`awgn`, `rayleigh`, `rician` |
| `--snr-db` | float | 20.0 | 目标 SNR (dB) |
| `--scs-khz` | int | 15 | 子载波间隔 (kHz) |
| `--n-fft` | int | 1024 | OFDM FFT 点数 |
| `--n-subcarriers` | int | 600 | 活跃子载波数 |
| `--dl-slots` | int | 72 | 下行时隙数 |
| `--ul-slots` | int | 2 | 上行时隙数 |
| `--delays` | str | "0,2,5" | 多径时延 (采样点) |
| `--powers-db` | str | "0,-3,-8" | 多径功率 (dB) |
| `--rician-k-db` | float | 6.0 | Rician K 因子 (dB) |
| `--doppler-hz` | float | 0.0 | 每径最大 Doppler 频移 (Hz) |
| `--seed` | int | 7 | 随机种子 |
| `--semantic` | choice | swinjscc | 语义模型：`swinjscc`, `fallback` |
| `--h264-crf` | int | 28 | H.264 CRF 质量参数 |
| `--channel-estimator` | choice | ls | 信道估计器：`ls`, `comm_foundation` |
| `--comm-foundation-checkpoint` | str | None | 复值通信基座模型 checkpoint |

## 核心模块说明

### config.py — 配置系统

使用冻结数据类 (`frozen dataclass`) 定义五组配置：

- **NRPhyConfig**：NR 物理层参数（子载波间隔、FFT大小、时隙数、SNR等）
- **ChannelConfig**：信道参数（类型、多径延迟/功率、Rician K因子）
- **SemanticConfig**：SwinJSCC 语义编解码配置
- **ConventionalConfig**：H.264+LDPC 传统方案配置
- **DemoConfig**：演示运行配置（图像路径、输出目录）

### channel.py — 信道模型

实现三种信道模型的统一接口 `MultipathChannel`：

| 信道类型 | 特点 |
|----------|------|
| AWGN | 加性高斯白噪声，无衰落 |
| Rayleigh | 瑞利衰落抽头延迟线(TDL)，NLoS场景 |
| Rician | 莱斯衰落TDL，包含LoS直射径分量 |

每次传输后返回 `ChannelOutput`，包含接收波形、冲激响应和测量的实际 SNR。

### semantic.py — 语义通信

封装 SwinJSCC (Swin Transformer-based Joint Source-Channel Coding) 模型：

- 支持真实 SwinJSCC 模型推理或确定性回落符号(fallback)
- 图像 → 复数符号 → QPSK 调制的完整编码管线
- 接收端 QPSK 解调 → 均衡 → SwinJSCC 解码 → 重建图像

### conventional.py — 传统方案

H.264 视频编码 + LDPC 信道编码的传统图像传输方案：

- H.264 帧内编码 (PyAV/FFmpeg)
- 规则 LDPC 码 (可配置码率和迭代次数)
- 重复编码增强可靠性

### ofdm.py — OFDM 调制

符合 5G NR Normal CP 规范的 OFDM 调制解调：

- 符号级可变 CP 长度（首/中符号长CP，其余短CP）
- ifftshift + IFFT + 加 CP 的完整调制链路
- 对应解调：去 CP + FFT + fftshift

### resource_grid.py — 资源网格

时频资源网格映射器，实现：

- 梳状导频插入（每 4 个 RE 插入 1 个导频）
- Gold 序列 QPSK 导频生成
- 语义数据和传统数据的正交资源分配

### dsp.py — 数字信号处理

- **ChannelEstimator**：LS 信道估计 + 频域线性插值 + 时域线性插值
- **CSI 结构化去噪**：块静态场景下跨 OFDM 符号平均，降低导频 LS 噪声
- **CSI 压缩**：优先使用 delay-domain sparse tap 压缩，频域降采样作为回退
- **CSI 解压**：delay-domain tap 恢复或频域升采样重建

### nodes.py — 网络节点

- **BaseStation (BS)**：构建下行发送（SwinJSCC编码 + 网格映射 + OFDM调制）
- **UserEquipment (UE)**：下行接收（OFDM解调 + 信道估计 + 均衡 + SwinJSCC解码）+ CSI 反馈构建

### simulation.py — 仿真主循环

`TDDPhysicalLayerSimulation` 编排完整的端到端流程：

1. 传统方案编码（H.264+LDPC）
2. BS 构建下行波形
3. 信道传输
4. UE 接收解码（语义 + 传统）
5. UE CSI 压缩反馈
6. 上行信道传输
7. BS CSI 恢复与质量评估
8. 生成可视化图表

### visualization.py — 可视化

自动生成以下图表：
- **星座图对比**：均衡前后 QPSK 星座
- **重建对比图**：原图、SwinJSCC 重建、H.264+LDPC 重建
- **时频资源网格图**：显示导频、语义数据、传统数据的分配
- **帧结构图**：TDD 上下行时隙配置

## 输出文件

仿真结果保存在 `outputs/{channel}_snr_{X}dB/` 目录下：

```
outputs/awgn_snr_20dB/
├── semantic_constellation.png    # 语义流星座图
├── traditional_constellation.png # 传统流星座图
├── reconstruction.png            # 图像重建对比
├── resource_grid.png             # 时频资源网格
├── frame_structure.png           # 帧结构示意图
├── semantic_reconstructed.png    # SwinJSCC 重建图像
├── traditional_reconstructed.png # H.264+LDPC 重建图像
├── run_summary.json              # 运行摘要 (JSON)
├── run_summary.md                # 运行摘要 (Markdown)
└── raw_console.log               # 完整控制台输出
```

## 典型运行结果

```
Channel=awgn, SNR=20.0 dB, SwinJSCC PSNR=28.24 dB, H.264+LDPC PSNR=34.16 dB, BS CSI NMSE=-19.43 dB.
```

- **SwinJSCC PSNR**：语义通信重建质量，在低SNR下通常优于传统方案
- **H.264+LDPC PSNR**：传统分离编码方案的参考基线
- **BS CSI NMSE**：基站恢复的 CSI 归一化均方误差（负值，越小越好）

## 项目特点

1. **模块化设计**：每个物理层组件独立成模块，易于扩展和替换
2. **5G NR 兼容**：遵循 NR Normal CP 帧结构、LS 信道估计、梳状导频等标准实践
3. **语义+传统双模对比**：同步运行 SwinJSCC 和 H.264+LDPC，便于公平比较
4. **TDD 闭环 CSI**：完整实现 UE 信道估计 → 压缩反馈 → BS 恢复的闭环流程
5. **丰富可视化**：自动生成星座图、资源网格、重建对比等图表
6. **可配置信道**：支持 AWGN、Rayleigh、Rician 三种信道及自定义多径参数
7. **确定性回落**：当 SwinJSCC 模型不可用时，自动使用确定性符号生成，确保代码可运行

## 参考文献

- Yang, K., et al. "SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding." arXiv, 2023.
- 3GPP TS 38.211: NR Physical channels and modulation.
- 3GPP TS 38.214: NR Physical layer procedures for data.

## License

本项目仅供学术研究使用。
