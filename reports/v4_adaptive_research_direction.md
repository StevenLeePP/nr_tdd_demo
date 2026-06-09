# 自适应研究方向报告

## 1. 实验目的

本阶段目标不是证明 `comm_foundation` 一定有效，而是建立一个结果驱动的闭环，判断当前 NR-TDD 语义通信链路里最值得继续投入的位置。约束保持不变：不做大型 Complex Transformer，不修改 SwinJSCC decoder，不做信道类型分类，不在 residual 仍未证明有效时继续盲目大规模训练。

## 2. 当前链路主要瓶颈

当前最明确的瓶颈是学习式 residual 训练路径，而不是模型容量。

- 已有正式 checkpoint 诊断：`residual_stats_trained.l2_norm = 0.0`，`H_hat` 与 `H_structured` 的差值 NMSE 为 `0.0`，`comm_foundation_trained_nmse = comm_foundation_untrained_nmse = 0.0048445659`。
- 新增小样本 overfit 诊断显示：完整 residual 路径可训练，32 个样本上允许 backbone + head 更新时，`H_hat NMSE` 从 `0.00331975` 降到 `0.000322384`，`residual_norm_after = 12.2779`。
- head-only 训练只能从 `0.00331975` 降到 `0.00308418`，不能强行 overfit，说明当前 head 表达或输入表征不足，不能只靠 1x1 residual head 解决。
- 修复训练输入为 `ls_smoothing` 后的小训练能产生非零 residual：`residual L2 = 1.49386`，但 smoke checkpoint 的 `comm_foundation_trained_nmse = 0.00487285`，仍差于 `ls_smoothing_nmse = 0.00484457`。

因此，主瓶颈排序为：

1. residual 训练目标/输入接口/模型选择逻辑；
2. CSI feedback 压缩质量；
3. 信道估计结构先验的进一步改进；
4. ReliabilityHead 是否可用于资源映射；
5. 运行时开销。

暂时没有证据说明 SwinJSCC decoder 是下一步最优改动点。

## 3. 当前最强 baseline

当前最强 baseline 是 `ls_smoothing`，也就是 LS + delay-domain denoise + block-static/time smoothing 这一类通信结构先验。

已有 rayleigh, SNR 10 dB, Doppler 0 Hz, pilot spacing 4 的 ablation：

| Estimator | CE NMSE dB | EVM dB | PSNR dB | BS CSI NMSE dB | Runtime ms |
|---|---:|---:|---:|---:|---:|
| ls | -14.17 | -8.21 | 27.30 | -5.54 | 0.00 |
| ls_smoothing | -25.66 | -9.63 | 27.57 | -5.88 | 4.87 |
| comm_foundation_untrained | -25.66 | -9.63 | 27.57 | -5.88 | 164.49 |
| comm_foundation_trained | -25.66 | -9.63 | 27.57 | -5.88 | 163.93 |

`gain_vs_baselines.csv` 已补生成。`comm_foundation_trained` 相对 `ls_smoothing` 的 CE/PSNR/EVM 增益接近 0，同时 runtime 增加约 159 ms。

## 4. 学习式模块是否带来净收益

当前结论：没有足够证据证明学习式 residual 已带来净收益。

- 相比结构先验：`comm_foundation_trained` 未超过 `ls_smoothing`。
- 相比 untrained：正式 checkpoint 中 trained 与 untrained 完全一致；smoke checkpoint 虽然 residual 非零，但还没有端到端收益。
- 端到端 PSNR：现有 ablation 中 trained 相比 smoothing 的 PSNR gain 为 `0.0 dB`。
- 运行时：学习式估计器相比 `ls_smoothing` 增加约 `159 ms`，在无收益时不合理。

较积极的信号是：完整 residual 路径可以在小样本上 overfit，说明不是计算图彻底断裂。问题更可能是训练输入与部署输入错位、best checkpoint 被 epoch0 identity-safe 截获、loss 与 structured baseline 的竞争关系不清晰。

## 5. 自适应决策结果

新增脚本输出：

- `outputs/adaptive_decision/next_action.json`
- `outputs/adaptive_decision/decision_report.md`
- `outputs/adaptive_decision_smoke_structured/next_action.json`
- `outputs/adaptive_decision_smoke_structured/decision_report.md`

正式 checkpoint 视角：

```text
next_action = debug_residual_training
reason = checkpoint diagnostic shows residual norm is approximately zero
```

修复训练输入并做 smoke checkpoint 后：

```text
next_action = do_not_expand_channel_estimator
reason = trained residual does not beat ls_smoothing on available gain rows
```

这两个结论并不冲突：第一层说明旧 checkpoint 没有真实 residual；第二层说明即便 residual 能动，当前也还不值得扩大神经信道估计器。

## 6. 下一步最优路线

从限定选项中选择：

```text
continue_residual_debug
```

选择原因：

- 正式 checkpoint residual 仍为 0，必须先修复训练路径和 checkpoint 选择逻辑。
- 小样本 full-path overfit 成功，说明继续 debug 有价值；但 head-only 不足，不能直接进入大规模训练。
- smoke structured 训练证明 residual 可非零，但没有超过 `ls_smoothing`，所以不能选择 `consider_complex_transformer`。
- CSI feedback 是潜在系统瓶颈，但在信道估计 residual 仍未稳定贡献前，不宜把主线完全迁移过去。

备选路线：

- 若后续 residual 稳定非零但仍不优于 `ls_smoothing`，切换到 `stop_neural_expansion_temporarily` 或 `learned_csi_feedback`。
- 若 ReliabilityHead 在更多 held-out 数据上稳定保持 high/low error ratio < 0.8，可切换到 `reliability_guided_mapping`。

## 7. 停止条件

停止继续扩大信道估计网络的条件：

- trained 与 untrained 输出再次长期一致；
- residual norm 接近 0；
- structured-input 训练后仍不能稳定超过 `ls_smoothing`；
- CE NMSE 提升不能传导到 EVM 或 SwinJSCC PSNR；
- runtime 增加明显但 PSNR 无收益；
- 预训练不优于 scratch，尤其在 1%、5%、10% 少样本与 held-out split 上不优。

允许进入更大模型或 Complex Transformer 的条件：

- `comm_foundation_trained` 相比 `ls_smoothing` 平均 CE NMSE 至少提升 1 dB；
- 或 SwinJSCC PSNR 稳定提升至少 0.1 dB；
- 并且 pretrain 在少样本/held-out 条件下优于 scratch；
- ReliabilityHead 能稳定区分高低可靠区域。

当前未满足这些条件。

## 8. 本阶段代码与结果文件

新增/更新脚本：

- `scripts/overfit_residual_head_debug.py`
- `scripts/adaptive_experiment_controller.py`
- `scripts/train_comm_foundation_model.py`
- `scripts/evaluate_comm_foundation_grid.py`
- `models/comm_foundation_model.py`

关键输出：

- `outputs/residual_overfit_debug/overfit_residual_debug.json`
- `outputs/residual_overfit_debug_backbone/overfit_residual_debug.json`
- `outputs/checkpoint_diagnostics/checkpoint_diagnostics.json`
- `outputs/checkpoint_diagnostics_smoke_structured/checkpoint_diagnostics.json`
- `outputs/grid_eval_ablation_rayleigh10/gain_vs_baselines.csv`
- `outputs/reliability_eval/reliability_metrics.csv`
- `outputs/adaptive_decision/next_action.json`

本阶段没有继续 Dataset-V1 正式大训练。理由是：虽然 full residual path 可以 overfit，但旧正式 checkpoint 为 zero residual，修复后的 smoke checkpoint 仍未超过结构先验。继续大训练前应先调整训练目标、checkpoint 选择标准、residual/head 表达能力和 structured-input 训练配置。
