# 通信基座模型工程探索报告

## 1. 当前 NR-TDD 语义链路

当前端到端数据流如下：

1. 图像输入由 `semantic.py` 中的 `SwinJSCCInterface` 读取。
2. SwinJSCC 编码器输出语义复数 IQ 符号。
3. `nodes.py` 中的 `BaseStation.build_downlink()` 将语义符号和 H.264+LDPC 符号映射到下行 OFDM 资源网格。
4. `resource_grid.py` 中的 `ResourceGridMapper` 在每个 OFDM 符号上插入梳状导频，导频与数据信号比例为 1:3。
5. `ofdm.py` 中的 `OFDMModem` 完成 IFFT 和随 OFDM 符号变化的 Normal CP 插入。
6. `channel.py` 中的 `MultipathChannel` 施加 AWGN、Rayleigh 或 Rician 多径信道。
7. UE 将接收波形解调回 `rx_grid`。
8. `dsp.py` 中的 `ChannelEstimator.estimate_slot()` 执行导频 LS 信道估计和二维插值。
9. MMSE 均衡器使用估计信道进行均衡。
10. 均衡后的语义符号送入现有 SwinJSCC decoder，完成图像语义重建。
11. UE 使用 `dsp.py` 中的 CSI 压缩函数压缩 `H_est`。
12. UE 将压缩 CSI bit 映射到上行资源网格，BS 接收后通过 `decompress_csi()` 和 `csi_nmse()` 评估恢复质量。

## 2. 当前信号张量

数据集导出 sanity check 使用 `Nsc=72`、`Nslot=1`、`Nsym=14`。

| 张量 | 含义 | sanity 数据集 shape | dtype |
|---|---|---:|---|
| `rx_grid` | 接收端 OFDM 资源网格 | `[100, 1, 72, 14]` | `complex64` |
| `pilot_obs` | 导频位置接收信号，非导频位置置零 | `[100, 1, 72, 14]` | `complex64` |
| `pilot_mask` | 梳状导频掩码 | `[100, 1, 72, 14]` | `bool` |
| `H_ls_grid` | LS + 插值得到的基线信道估计 | `[100, 1, 72, 14]` | `complex64` |
| `H_true` | 由仿真信道冲激响应得到的真实频域响应 | `[100, 1, 72, 14]` | `complex64` |
| `equalized_symbols` | 均衡后的数据符号 | `[100, 756]` | `complex64` |
| `semantic_tx_symbols` | 发送端语义/合成 payload 符号 | `[100, 756]` | `complex64` |

`H_true` 可以在仿真中获得。默认 block-static 信道下，`MultipathChannel.transmit()` 返回时域冲激响应，`impulse_response_to_grid()` 将其转换为激活子载波上的频域信道响应；当启用 Doppler 时，信道对象会额外给出逐 OFDM 符号的 `frequency_response_grid`。

指标位置如下：

- 图像 PSNR：`simulation.py` 中语义分支和传统分支计算。
- CSI 压缩与 BS 侧恢复 NMSE：`simulation.py` 中 `_evaluate_csi_feedback()` 计算。
- 信道估计 NMSE：`simulation.py` 中 `csi_nmse(H_true, H_est)` 计算。
- 语义均衡后 EVM：`simulation.py` 中 `_evm()` 计算。
- 训练 NMSE loss：`models/comm_foundation_model.py` 中 `nmse_loss()` 计算。
- 学习式估计器推理耗时：`UserEquipment.receive_downlink()` 累加，并写入 `channel_estimation_quality`。

## 3. 数据集导出

新增脚本：

```bash
python scripts/export_comm_foundation_dataset.py \
  --num_samples 100 \
  --snr_min 0 \
  --snr_max 30 \
  --doppler_min 0 \
  --doppler_max 0 \
  --channel_mode mixed \
  --output_path outputs/comm_foundation_sanity_100.npz \
  --seed 2026
```

脚本输出 `.npz` 数组文件和一个 sidecar JSON 摘要。`channel_mode` 只作为数据生成控制项，不作为信道类型分类任务标签。`doppler` 也是生成控制变量；当 Doppler 非零时，仿真会施加轻量的逐路径时变相位，并导出对应逐符号 `H_true`。

这符合当前研究目标：模型不应服务于 AWGN/Rayleigh/Rician 分类，而应服务于信道估计、CSI 反馈、均衡辅助和语义重建增强等实际链路任务。

sanity 数据集结果：

- 样本数：`100`
- 导出摘要中的 LS baseline 平均 NMSE：`0.0504`
- shape 和 dtype 已在 `comm_foundation_sanity_100.json` 中验证。

## 4. 复值 Backbone 设计

新增模块：

- `models/complex_layers.py`
- `models/comm_foundation_model.py`

当前最小实现采用 `[B, 2*C, F, T]` 表示复数张量，其中前 `C` 个通道为实部，后 `C` 个通道为虚部。复值层内部遵循复数乘法规则：

```text
Yr = Wr * Xr - Wi * Xi
Yi = Wr * Xi + Wi * Xr
```

已实现：

- `ComplexConv2d`
- `ComplexLinear`
- `ComplexReLU`
- `ComplexLayerNorm`
- `ComplexCommunicationBackbone`

当前估计器已经改为 residual-safe 结构：

```text
H_hat = H_structured + ComplexResidualHead(Z_comm)
```

其中 `H_structured` 是经过通信结构先验处理后的 LS 估计。残差头零初始化，因此未训练或训练不足的 checkpoint 不会用随机复值特征覆盖 LS，而是从一个非破坏性的估计器开始。

通信潜空间输出为：

```text
Z_comm = ComplexCommunicationBackbone(H_ls_grid)
```

默认 sanity 模型中，`Z_comm` 的 shape 为 `[B, 2*hidden_complex_channels, F, T]`；当 `hidden_complex_channels=16` 时，即 `[B, 32, F, T]`。

## 5. 参考论文对设计的启发

PDF `Unveiling the Power of Complex-Valued Transformers in Wireless Communications` 对当前方向的主要启发如下：

- 无线基带信号和信道响应天然是复值数据，应尽量原生建模幅度和相位，而不是把实部和虚部当作无关的实值通道。
- 论文从理论和实验上讨论了复值神经网络的优势，并提出了包含 embedding、encoding、decoding 和 output projection 的复值 Transformer 范式。
- 论文覆盖的代表性无线任务包括信道估计、联合导频/反馈量化/预编码设计等，与当前通信基座模型目标一致。

对本工程的含义：

- 当前 `ComplexCommunicationBackbone` 是为了跑通最小闭环而保留的轻量 Complex CNN。
- 模块接口刻意保持可替换，后续可以升级为 Complex Transformer：

```text
OFDM RE/patch 复值 embedding
→ 时频 token 上的复值 self-attention
→ Z_comm
→ 多任务 head
```

当前 task heads 和训练脚本在替换 backbone 后仍应复用。

## 6. 当前任务头

已实现：

- `ChannelEstimationHead`：输出 `H_hat`。

已预留：

- `CSIFeedbackHead`：后续用于学习式 CSI 压缩和重建，替代手工量化反馈。
- `ReliabilityHead`：后续输出时频可靠性图，用于语义符号保护或资源分配。
- `SemanticAssistHead`：后续输出辅助 SwinJSCC decoder 的条件特征，不直接替代图像 decoder。

这些任务头位于 `comm_foundation_model.py`。

## 7. 训练脚本

新增脚本：

```bash
python scripts/train_comm_foundation_model.py \
  --dataset_path outputs/comm_foundation_sanity_100.npz \
  --output_dir outputs/comm_foundation_ckpt_residual_safe \
  --epochs 1 \
  --batch_size 16 \
  --lr 0.001 \
  --use_masked_csi \
  --use_denoising
```

支持的训练目标：

```text
L_pretrain = lambda_ce * L_channel_estimation
           + lambda_mask * L_masked_csi
           + lambda_denoise * L_denoising_csi
```

训练脚本会先保存 epoch 0 的 identity-safe checkpoint，然后只在验证 NMSE 改善时替换 best checkpoint。这避免弱模型盲目替代 LS 后破坏均衡和语义重建。

当前用于端到端验证的本地 checkpoint 为：

- `outputs/comm_foundation_ckpt_residual_safe/best_comm_foundation_channel_estimator.pt`

## 8. 仿真链路接入

默认 LS 链路保持不变：

```bash
python run_demo.py --channel-estimator ls
```

学习式估计器作为可选路径启用：

```bash
python run_demo.py \
  --channel-estimator comm_foundation \
  --comm-foundation-checkpoint outputs/comm_foundation_ckpt_residual_safe/best_comm_foundation_channel_estimator.pt
```

启用后流程如下：

1. UE 仍先从导频计算 `H_ls_grid`。
2. 当 `doppler_hz=0` 时，`LearnedChannelEstimator` 先对一个 slot 内的 CSI 进行跨 OFDM 符号平均，利用 TDD block-static 信道相干先验降低 LS 噪声。
3. 复值 residual 模型在结构化估计基础上输出 `H_hat`。
4. `H_hat` 替代原始 LS 插值结果参与 MMSE 均衡。
5. SwinJSCC decoder 保持不变。
6. 日志输出信道估计 NMSE、语义 EVM、图像 PSNR 等指标。
7. 使用 learned estimator 时，日志会额外记录模型推理耗时。

CSI 反馈现在优先使用 delay-domain sparse tap 压缩：

```text
H_est → delay-domain taps over time segments → scalar quantization → UL QPSK feedback
```

如果 delay-domain 方案无法适配 UL 容量，则回退到旧的 frequency-stride 反馈。

## 9. 初步实验结果

Rayleigh 20 dB，LS 估计器：

- SwinJSCC PSNR：约 `28.10 dB`
- H.264+LDPC PSNR：约 `34.16 dB`
- 信道估计 NMSE：约 `-24.17 dB`
- delay-domain 反馈后 BS 恢复 CSI NMSE：约 `-11.69 dB`

Rayleigh 20 dB，residual-safe foundation checkpoint + block-static CSI smoothing：

- SwinJSCC PSNR 提升到约 `28.17 dB`
- 信道估计 NMSE 提升到约 `-35.62 dB`
- delay-domain 反馈后 BS 恢复 CSI NMSE 约 `-11.91 dB`

解释：第一阶段有效增益来自“通信结构先验 + 复值 residual 模型接口”的组合。神经 residual 本身还需要更大规模预训练，才能成为真正独立的学习式改进项；但当前 `comm_foundation` 路径已经不再破坏链路，并在默认 Rayleigh 20 dB 场景下改善了信道估计、反馈质量和语义重建。

Rayleigh 10 dB 额外验证：

| 估计器 | SwinJSCC PSNR | CE NMSE | 语义 EVM | BS CSI NMSE |
|---|---:|---:|---:|---:|
| LS | `27.30 dB` | `-14.17 dB` | `-8.21 dB` | `-5.54 dB` |
| comm_foundation | `27.57 dB` | `-25.66 dB` | `-9.63 dB` | `-5.88 dB` |

## 10. 下一步建议

1. 扩大数据集覆盖范围，包括 SNR、Doppler、delay profile、导频密度和反馈预算。
2. 延长训练并在 held-out SNR 和信道 profile 上与 LS/MMSE baseline 对比。
3. 将轻量 Complex CNN 替换为参考论文启发的 Complex Transformer：
   - OFDM RE block 的复值 patch/token embedding；
   - 时频 token 上的复值 self-attention；
   - 面向信道估计和 CSI 反馈的复值 projection heads。
4. 实现真正的 `CSIFeedbackHead`，替代当前 delay-domain taps + scalar quantization。
5. 实现 `ReliabilityHead`，将可靠性图用于资源分配或 SwinJSCC 符号保护。
6. 在保持 SwinJSCC decoder 不变的前提下加入 semantic-aware loss。
7. 重点评估 Doppler、稀疏导频、低 SNR 和强 CSI 压缩压力下的泛化能力。
