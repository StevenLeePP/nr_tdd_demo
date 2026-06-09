# v2 复值通信基座模型下一阶段实验报告

## 1. 目标

当前最小闭环已经证明 `comm_foundation` 可以接入 NR-TDD 语义通信链路，并在默认 Rayleigh 场景下改善 CE NMSE、EVM 和 SwinJSCC PSNR。但这些收益可能来自两类来源：

1. 通信结构先验，例如 block-static smoothing；
2. 复值神经网络 residual 在 `Z_comm` 中学到的可迁移修正能力。

下一阶段的核心目标不是立即升级 Complex Transformer，也不是修改 SwinJSCC decoder，而是通过消融、批量评估、少样本训练、跨条件测试和多任务头来验证：复值 backbone 是否真的学到了可迁移通信潜空间 `Z_comm`。

## 2. Estimator 消融定义

`run_demo.py --channel-estimator` 现在支持四种主要模式：

| 模式 | 定义 | 是否使用结构先验 | 是否使用神经网络 |
|---|---|---:|---:|
| `ls` | 原始 LS + 插值 | 否 | 否 |
| `ls_smoothing` | LS 后只做 block-static smoothing / structured estimate | 是 | 否 |
| `comm_foundation_untrained` | structured estimate + 未训练 residual-safe 复值模型 | 是 | 是，但 residual 零初始化 |
| `comm_foundation_trained` | structured estimate + checkpoint 中训练过的复值 residual 模型 | 是 | 是 |

旧参数 `comm_foundation` 仍作为 `comm_foundation_trained` 的兼容别名保留。

四种模式统一输出：

- CE NMSE；
- semantic EVM；
- SwinJSCC PSNR；
- BS 侧 CSI feedback NMSE；
- channel estimator runtime。

示例：

```bash
python run_demo.py \
  --channel rayleigh \
  --snr-db 10 \
  --channel-estimator ls_smoothing

python run_demo.py \
  --channel rayleigh \
  --snr-db 10 \
  --channel-estimator comm_foundation_trained \
  --comm-foundation-checkpoint outputs/comm_foundation_ckpt_residual_safe/best_comm_foundation_channel_estimator.pt
```

## 3. 批量评估脚本

新增脚本：

```bash
python scripts/evaluate_comm_foundation_grid.py \
  --estimators ls,ls_smoothing,comm_foundation_untrained,comm_foundation_trained \
  --snrs 10,20 \
  --dopplers 0,50,100 \
  --channels rayleigh,rician \
  --pilot-spacings 4,6,8 \
  --checkpoint outputs/comm_foundation_ckpt_residual_safe/best_comm_foundation_channel_estimator.pt \
  --output-dir outputs/grid_eval
```

输出：

- `metrics.csv`
- `summary.md`

脚本会为每个 case 创建独立输出目录，避免不同 SNR、Doppler 或 pilot spacing 的结果互相覆盖。

## 4. Dataset-V1 设计

`scripts/export_comm_foundation_dataset.py` 现在支持 `--dataset_version v1`：

```bash
python scripts/export_comm_foundation_dataset.py \
  --dataset_version v1 \
  --num_samples 20000 \
  --val_samples 2000 \
  --heldout_samples 2000 \
  --snr_min 0 \
  --snr_max 30 \
  --doppler_min 0 \
  --doppler_max 100 \
  --channel_mode mixed \
  --output_dir outputs/dataset_v1
```

Dataset-V1 生成规则：

- Rayleigh + Rician mixed；
- SNR 在 `0~30 dB` 随机采样；
- Doppler 在 `0~100 Hz` 随机采样；
- delay profile 随机扰动；
- Rician K 因子在训练范围内随机扰动；
- 信道模式只用于数据生成，不作为分类任务。

输出 split：

| split | 目的 |
|---|---|
| `train.npz` | 主训练集 |
| `val.npz` | 同分布验证集 |
| `test_unseen_snr.npz` | 未见 SNR 范围 |
| `test_unseen_doppler.npz` | 未见 Doppler 范围 |
| `test_unseen_delay.npz` | 未见 delay profile |
| `test_unseen_rician_k.npz` | 未见 Rician K 范围 |
| `manifest.json` | 数据集摘要 |

本轮 smoke test 已用极小样本验证 v1 导出链路，所有 split 都能生成 `.npz` 和 `.json` 摘要。

## 5. 预训练策略对比

`scripts/train_comm_foundation_model.py` 新增训练策略：

| 策略 | 含义 |
|---|---|
| `scratch_ce` | 从零训练，只做信道估计 |
| `pretrain_then_finetune` | 先 masked CSI + denoising CSI 预训练，再微调信道估计 |
| `joint_pretrain_ce` | masked、denoising、channel estimation 联合训练 |

少样本比例：

```bash
--sample_fraction 0.01
--sample_fraction 0.05
--sample_fraction 0.10
--sample_fraction 1.0
```

held-out 测试集评估：

```bash
python scripts/train_comm_foundation_model.py \
  --dataset_path outputs/dataset_v1/train.npz \
  --eval_dataset_paths outputs/dataset_v1/test_unseen_snr.npz,outputs/dataset_v1/test_unseen_doppler.npz \
  --training_strategy pretrain_then_finetune \
  --sample_fraction 0.05 \
  --use_reliability \
  --output_dir outputs/ckpt_pretrain_5pct
```

训练脚本会记录：

- validation NMSE；
- validation LS baseline；
- held-out split NMSE；
- ReliabilityHead 相关指标。

## 6. ReliabilityHead 设计

`models/comm_foundation_model.py` 中的 `ReliabilityHead` 已实现为最小可训练版本：

```text
Z_comm → ReliabilityHead → reliability_map
```

输出：

- shape：`[B, 1, F, T]`；
- 范围：`[0, 1]`；
- 与时频资源网格对齐。

第一阶段不接入 SwinJSCC decoder，不改变图像解码路径。训练标签来自信道估计误差：

```text
error_map = |H_hat - H_true|^2
reliability_target = exp(-error_map / tau)
```

loss：

```text
L_reliability = MSE(reliability_map, reliability_target)
```

当前评估指标：

- `reliability_error_corr`：可靠性图与误差图的相关性；
- `high_reliability_error`：高可靠区域均衡/信道误差；
- `low_reliability_error`：低可靠区域均衡/信道误差；
- `reliability_target_mse`：可靠性预测与目标图的 MSE。

smoke test 已验证 ReliabilityHead 的 loss 和指标能正常写入 `training_summary.json`。

## 7. 初步消融结果

小网格配置：

```text
channel = rayleigh
SNR = 10 dB
Doppler = 0 Hz
pilot_spacing = 4
semantic = swinjscc
```

结果来自 `outputs/grid_eval_ablation_rayleigh10/metrics.csv`：

| Estimator | PSNR | CE NMSE | EVM | BS CSI NMSE | Runtime |
|---|---:|---:|---:|---:|---:|
| `ls` | `27.30 dB` | `-14.17 dB` | `-8.21 dB` | `-5.54 dB` | `0.000 ms` |
| `ls_smoothing` | `27.57 dB` | `-25.66 dB` | `-9.63 dB` | `-5.88 dB` | `4.870 ms` |
| `comm_foundation_untrained` | `27.57 dB` | `-25.66 dB` | `-9.63 dB` | `-5.88 dB` | `164.490 ms` |
| `comm_foundation_trained` | `27.57 dB` | `-25.66 dB` | `-9.63 dB` | `-5.88 dB` | `163.931 ms` |

阶段性解释：

- 当前收益主要来自 `ls_smoothing` 这类通信结构先验。
- `comm_foundation_untrained` 与 `comm_foundation_trained` 结果相同，说明当前 residual 网络尚未证明额外学习收益。
- 因此下一步必须用 Dataset-V1、少样本比例和 held-out 条件验证 residual backbone 是否真正学习了可迁移 `Z_comm`。

## 8. 是否升级 Complex Transformer 的判断依据

暂时不优先实现大型 Complex Transformer。只有当下面证据成立时，再升级 backbone：

1. `comm_foundation_trained` 在相同 structured estimate 输入下优于 `comm_foundation_untrained`；
2. `pretrain_then_finetune` 在 1%、5%、10% 少样本下优于 `scratch_ce`；
3. 预训练模型在未见 SNR、Doppler、delay profile 或 Rician K 范围上更稳；
4. `Z_comm` 能同时支撑 ChannelEstimationHead 和 ReliabilityHead；
5. ReliabilityHead 输出与真实 error map 有稳定负相关，并能区分高低可靠区域。

如果这些证据不成立，优先改进数据生成、损失设计和结构化估计接口，而不是直接堆大模型。
