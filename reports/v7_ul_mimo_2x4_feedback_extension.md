# 上行 2x4 MIMO 反馈链路扩展报告

## 1. 本次修改目标

本次完成两个修复/扩展：

1. 修复 `--semantic fallback` 时 `psnr_db=None` 导致 `write_logs()` 中 `.2f` 格式化抛 `TypeError` 的问题；
2. 在已完成的下行 4x4 MIMO 基础上，增加上行 2x4 MIMO 反馈链路。

默认 SISO 路径仍保持：

```bash
python run_demo.py --channel-estimator ls
```

## 2. 新增上行 MIMO 参数

新增参数：

```bash
--ul-num-tx-antennas 2
--ul-num-rx-antennas 4
--ul-array-size 1x4
```

含义：

- UE 端上行使用 2 个发射天线；
- BS 端上行使用 4 个接收天线；
- 上行表现为 2 层空间流；
- 阵列仍按 1x4 ULA 记录。

## 3. 上行 2x4 MIMO 张量

上行反馈发送波形：

```text
UL X_tx shape = [N_ul_tx, samples]
```

上行反馈接收波形：

```text
UL Y_rx shape = [N_ul_rx, samples]
```

上行 MIMO 信道频响：

```text
UL H_true shape = [B_ul, N_ul_rx, N_ul_tx, N_sc, N_sym]
```

本次 smoke 中：

```text
UL X_tx shape = [2, 30720]
UL Y_rx shape = [4, 30720]
UL H_true shape = [2, 4, 2, 600, 14]
```

其中第一个维度 `2` 是上行 slot 数，不是天线数。

## 4. 当前上行导频设计

上行也采用 Tx 间错开的 FDM comb pilot：

- 每个 OFDM 符号都有导频；
- UE 的 2 个 Tx 天线使用不同 comb offset；
- UE Tx0 和 Tx1 的导频不会混叠；
- BS 端可以分别估计每个 BS Rx 与 UE Tx 的信道。

因此 BS 侧上行 LS 信道估计得到：

```text
H_ul_ls[slot, bs_rx, ue_tx, subcarrier, symbol]
```

## 5. 当前上行 Equalizer

上行反馈采用 2 流空间复用：

```text
y = H x + n
```

其中：

- `x` 是 UE 两个发射天线上的两条反馈符号流；
- `H` 是 `4x2` MIMO 信道矩阵；
- `y` 是 BS 四个接收天线观测。

BS 侧使用 per-RE MIMO MMSE equalizer：

```text
x_hat = (H^H H + sigma^2 I)^(-1) H^H y
```

输出 shape：

```text
[N_ul_tx, N_sc, N_sym]
```

随后按 Tx0、Tx1 的映射顺序恢复 QPSK 反馈符号流，再解调为 CSI feedback bitstream。

## 6. Fallback PSNR bug 修复

问题：

```text
reconstructed.get("psnr_db") == None
```

时，旧代码：

```python
f"{reconstructed.get('psnr_db'):.2f}"
```

会抛：

```text
TypeError: unsupported format string passed to NoneType.__format__
```

修复：

- 新增 `format_metric()`；
- `None` 输出为 `N/A`；
- 数值正常按指定精度格式化。

验证命令：

```bash
python run_demo.py \
  --channel-estimator ls \
  --semantic fallback \
  --channel rayleigh \
  --snr-db 10 \
  --h264-crf 51 \
  --output-dir outputs/fallback_psnr_none_smoke
```

结果：

```text
SwinJSCC PSNR=N/A
```

并且日志正常写出。

## 7. DL 4x4 + UL 2x4 Smoke 测试

测试命令：

```bash
python run_demo.py \
  --channel-estimator ls \
  --semantic swinjscc \
  --channel rayleigh \
  --snr-db 10 \
  --h264-crf 51 \
  --num-tx-antennas 4 \
  --num-rx-antennas 4 \
  --array-type ula \
  --array-size 1x4 \
  --ul-num-tx-antennas 2 \
  --ul-num-rx-antennas 4 \
  --ul-array-size 1x4 \
  --output-dir outputs/mimo_smoke_dl4x4_ul2x4_ls
```

结果：

| 指标 | 值 |
|---|---:|
| SwinJSCC PSNR | `28.39 dB` |
| Semantic EVM | `-20.64 dB` |
| DL CE NMSE | `-18.02 dB` |
| BS recovered CSI NMSE | `-9.38 dB` |
| Feedback BER | `0.002172` |

关键 shape：

| 张量 | Shape |
|---|---|
| DL `X_tx` | `[72, 4, 600, 14]` |
| DL `Y_rx` | `[72, 4, 600, 14]` |
| DL `H_true/H_est` | `[72, 4, 4, 600, 14]` |
| UL `X_tx` | `[2, 30720]` |
| UL `Y_rx` | `[4, 30720]` |
| UL `H_true` | `[2, 4, 2, 600, 14]` |
| UL equalizer | `spatial_stream_mmse` |

与上一版 DL 4x4 + UL SISO smoke 相比，BS recovered CSI NMSE 从约 `-2.02 dB` 改善到 `-9.38 dB`。这说明上行 2 流 + BS 4Rx MMSE 合并对 CSI feedback 链路有明显帮助。

## 8. 当前限制

当前仍是最小可运行版本：

- 上行 MIMO 已支持 2 流反馈，但还没有做自适应调制编码；
- 上行导频是 comb offset 正交，没有加入码域正交或 DMRS-like 正交序列；
- 上行 MIMO 信道仍是独立 Rx-Tx TDL，暂未建模 ULA 角度扩展和空间相关；
- CSI feedback 仍压缩代表性下行子信道，尚未扩展到完整 4x4 DL CSI 的 learned feedback。

下一步如果继续做上行，应优先扩展完整 MIMO CSI feedback，而不是继续只反馈 `rx0-tx0` 代表子信道。
