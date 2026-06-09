# 低信噪比与多普勒信道估计压力测试

## 1. 目的

本阶段把信道估计测试从常规 SNR 区间下压到更低信噪比，并引入多普勒、稀疏导频、随机 delay profile 和 Rician K 因子扰动。目标是寻找机器学习信道估计可能优于传统结构先验的条件，而不是预设某一类算法必然获胜。

## 2. 新增测试入口

新增脚本：

```bash
python scripts/evaluate_channel_estimator_stress.py
```

默认 stress 范围：

- SNR: `-10,-5,0,5,10 dB`
- Doppler: `0,30,60,100,150 Hz`
- Channel: `rayleigh,rician`
- Pilot spacing: `4,6,8,12`
- Delay profile: `mixed`
- Rician K: `-3 到 12 dB`
- Delay-domain denoise taps: `16`

输出：

- `outputs/low_snr_doppler_ce_stress/metrics.csv`
- `outputs/low_snr_doppler_ce_stress/sample_metrics.csv`
- `outputs/low_snr_doppler_ce_stress/gain_vs_baselines.csv`
- `outputs/low_snr_doppler_ce_stress/summary.md`

## 3. 当前 smoke 结果

为避免直接启动大网格，本次先运行一个小规模 smoke：

```bash
python scripts/evaluate_channel_estimator_stress.py \
  --snrs -5,0 \
  --dopplers 0,100 \
  --channels rayleigh \
  --pilot-spacings 8 \
  --samples-per-case 3 \
  --output-dir outputs/low_snr_doppler_ce_stress_smoke
```

该 smoke 用于验证低 SNR、多普勒、随机 delay profile 下评估链路可以跑通。完整网格可以在确认训练 checkpoint 有效后再运行。

正式 zero-residual checkpoint 结果：

| 对比 | 平均 CE gain |
|---|---:|
| `ls_smoothing` vs `ls` | `5.046 dB` |
| `comm_foundation_trained` vs `ls` | `5.046 dB` |
| `comm_foundation_trained` vs `ls_smoothing` | `0.000 dB` |
| `comm_foundation_trained` vs `comm_foundation_untrained` | `0.000 dB` |

这说明低 SNR 下结构先验仍然明显强于 raw LS，但正式 checkpoint 没有证明学习式 residual 的额外收益。

使用 structured-input smoke checkpoint 的同一 stress 结果：

| 对比 | 平均 CE gain | 正增益比例 |
|---|---:|---:|
| `ls_smoothing` vs `ls` | `5.046 dB` | `1.00` |
| `comm_foundation_trained` vs `ls` | `5.079 dB` | `1.00` |
| `comm_foundation_trained` vs `ls_smoothing` | `0.033 dB` | `0.75` |
| `comm_foundation_trained` vs `comm_foundation_untrained` | `0.033 dB` | `0.75` |

这个结果是一个弱积极信号：当 residual checkpoint 确实非零时，在低 SNR stress 条件下能看到极小的平均收益，但幅度只有 `0.033 dB`，且 `-5 dB / 100 Hz` 条件下为负。因此它只能支持继续针对低 SNR/Doppler 分布训练与诊断，不能支持扩展到大型 Complex Transformer。

关键输出：

- `outputs/low_snr_doppler_ce_stress_smoke/summary.md`
- `outputs/low_snr_doppler_ce_stress_smoke_structured_ckpt/summary.md`

## 4. 判断规则

机器学习方法只有同时满足以下条件，才认为真的比传统方法强：

- `comm_foundation_trained` 相比 `ls_smoothing` 的 CE NMSE gain 为正；
- `comm_foundation_trained` 相比 `comm_foundation_untrained` 的 CE NMSE gain 为正；
- 低 SNR 和高 Doppler 条件下仍稳定，而不是只在单个随机样本上偶然胜出；
- runtime 增加可接受；
- 后续端到端测试中，CE NMSE 收益能传导到 EVM 或 SwinJSCC PSNR。

当前已知正式 checkpoint 的 residual 仍为 0，因此不能用它证明机器学习方法强于传统方法。低 SNR 压力测试的价值在于暴露结构先验失效区域，并为下一轮 residual 训练提供更有信息量的数据分布。
