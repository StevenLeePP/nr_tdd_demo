# v3 复值通信基座模型系统实验计划与结果汇总

## 1. 实验目的

本阶段目标是系统区分 `comm_foundation` 的收益来源：

1. 通信结构先验，例如 `ls_smoothing`；
2. 复值 residual backbone 在 `Z_comm` 中学到的可迁移修正能力。

本阶段暂不实现 Complex Transformer，不修改或替换 SwinJSCC decoder，不加入信道类型分类任务。

## 2. Estimator 消融结果

计划使用 `scripts/evaluate_comm_foundation_grid.py` 覆盖：

- estimator：`ls`、`ls_smoothing`、`comm_foundation_untrained`、`comm_foundation_trained`
- channel：`rayleigh`、`rician`
- SNR：`0,5,10,15,20,25,30`
- Doppler：`0,30,60,100`
- pilot spacing：`4,6,8`

输出：

- `metrics.csv`
- `summary.md`
- `gain_vs_baselines.csv`

重点比较：

- `comm_foundation_trained` vs `ls_smoothing`
- `comm_foundation_trained` vs `comm_foundation_untrained`

当前状态：脚本已支持 gain 统计，完整大网格尚未运行。

## 3. Checkpoint 诊断结果

计划新增 `scripts/diagnose_comm_foundation_checkpoint.py`，用于检查：

- residual 输出均值、方差、最大值、L2 norm；
- `H_hat` 与 `H_structured` 的差值 NMSE；
- trained checkpoint 与 epoch0 / untrained checkpoint 的参数差异；
- `Z_comm` 的均值和方差；
- 同一 batch 上的 `ls_smoothing`、`comm_foundation_untrained`、`comm_foundation_trained` NMSE。

当前状态：待实现。

## 4. Dataset-V1 训练结果

计划使用 Dataset-V1：

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

训练策略：

- `scratch_ce`
- `pretrain_then_finetune`
- `joint_pretrain_ce`

样本比例：

- `0.01`
- `0.05`
- `0.10`
- `1.0`

输出：

- `outputs/training_comparison/training_results.csv`
- `outputs/training_comparison/training_summary.md`

当前状态：训练脚本已支持三种策略、少样本比例和 held-out eval；正式 Dataset-V1 大规模训练尚未运行。

## 5. 少样本和 Held-Out 泛化结果

需要评估：

- `val.npz`
- `test_unseen_snr.npz`
- `test_unseen_doppler.npz`
- `test_unseen_delay.npz`
- `test_unseen_rician_k.npz`

关键问题：

1. 预训练是否优于 `scratch_ce`？
2. 预训练优势是否在 `1%`、`5%`、`10%` 少样本下更明显？
3. 预训练是否提升 held-out 泛化？
4. 当前结果是否足以证明 `Z_comm` 具有可迁移通信潜空间特性？

当前状态：待正式训练后填表。

## 6. ReliabilityHead 结果

当前训练脚本已支持 `--use_reliability`，需要进一步新增独立评估脚本：

- `scripts/evaluate_reliability_head.py`

计划输出：

- `reliability_metrics.csv`
- 可选 reliability heatmap

统计指标：

- `reliability_error_corr`
- `high_reliability_error`
- `low_reliability_error`
- `high_low_error_ratio`
- `reliability_target_mse`

当前状态：训练脚本已有最小 ReliabilityHead loss 和评估字段；独立评估脚本待实现。

## 7. 端到端语义链路复测

选择训练效果最好的 1 到 2 个 checkpoint 后，重新接入 `run_demo.py` 做端到端测试。

测试条件：

- channel：`rayleigh`、`rician`
- SNR：`5,10,20`
- Doppler：`0,60,100`
- pilot spacing：`4,8`

对比：

- `ls`
- `ls_smoothing`
- `comm_foundation_untrained`
- `comm_foundation_trained`

输出：

- `end_to_end_metrics.csv`
- `end_to_end_summary.md`

当前状态：待正式 checkpoint 训练完成后复测。

## 8. Go / No-Go 判断

Go 条件：

- `comm_foundation_trained` 相比 `ls_smoothing` 平均 CE NMSE 至少提升 `1 dB`；或
- SwinJSCC PSNR 稳定提升至少 `0.1 dB`；或
- 预训练在 `5%` 数据下明显优于 `scratch_ce`；
- 并且 ReliabilityHead 能稳定区分高低可靠区域。

No-Go 条件：

- trained 与 untrained 输出长期一致；
- residual norm 接近 0；
- 预训练不优于 scratch；
- held-out 条件下性能不稳定；
- 端到端 PSNR 无法受益。

若 No-Go，不实现 Complex Transformer，优先修复训练目标、residual scale、数据归一化、checkpoint 加载和模型接口。

## 9. 下一步建议

1. 实现 checkpoint 诊断脚本，先确认 trained residual 是否真的非零。
2. 运行中等规模 Dataset-V1，并完成三种训练策略和四个样本比例的对比。
3. 实现 ReliabilityHead 独立评估脚本。
4. 使用最优 checkpoint 进行端到端语义链路复测。
5. 根据 Go / No-Go 条件判断是否进入 Complex Transformer 阶段。
