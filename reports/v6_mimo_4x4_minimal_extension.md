# 4x4 MIMO 最小可运行扩展报告

## 1. 修改范围

本次把当前 SISO NR-TDD 语义通信链路扩展为可选 4x4 MIMO 下行链路。默认参数仍是 SISO，因此以下命令保持可运行：

```bash
python run_demo.py --channel-estimator ls
```

新增参数：

```bash
--num-tx-antennas 4
--num-rx-antennas 4
--array-type ula
--array-size 1x4
```

主要修改文件：

- `config.py`
- `run_demo.py`
- `channel.py`
- `ofdm.py`
- `resource_grid.py`
- `dsp.py`
- `nodes.py`
- `simulation.py`
- `learned_estimator.py`
- `models/comm_foundation_model.py`
- `scripts/export_comm_foundation_dataset.py`

## 2. SISO 与 4x4 MIMO 张量维度

SISO 原路径：

| 张量 | Shape |
|---|---|
| `X_tx` | `[B, N_sc, N_sym]` |
| `Y_rx` | `[B, N_sc, N_sym]` |
| `H_true` | `[B, N_sc, N_sym]` |
| `H_ls / H_est` | `[B, N_sc, N_sym]` |

4x4 MIMO 下行：

| 张量 | Shape |
|---|---|
| `X_tx` | `[B, N_tx, N_sc, N_sym]` |
| `Y_rx` | `[B, N_rx, N_sc, N_sym]` |
| `H_true` | `[B, N_rx, N_tx, N_sc, N_sym]` |
| `H_ls / H_est` | `[B, N_rx, N_tx, N_sc, N_sym]` |

本项目代码中 `B` 对应 slot/batch 维。Dataset 导出会保留 sample 和 slot 两层，例如 smoke 输出：

```text
H_true: [sample, slot, N_rx, N_tx, N_sc, N_sym]
        [2, 1, 4, 4, 72, 14]
```

## 3. 为什么 4x4 MIMO 不是 16 个发射天线

4x4 MIMO 表示：

- BS 端有 `4` 个发射天线；
- UE 端有 `4` 个接收天线；
- 每个接收天线都会看到每个发射天线到它的信道。

所以信道矩阵是 `N_rx x N_tx = 4 x 4`，共有 16 条 Rx-Tx 空间子信道：

```text
H[rx, tx, subcarrier, symbol]
```

这 16 个元素是空间信道矩阵的元素，不是 16 个发射天线。发射天线数量仍然是 4。

## 4. 当前导频设计

第一版采用 FDM orthogonal comb pilot：

- 每个 OFDM 符号都存在导频；
- 不同 Tx 天线使用错开的 comb 子集合；
- 为避免 4 个 Tx 的导频占满全部 RE，MIMO 下每个 Tx 使用 `pilot_spacing * N_tx` 的子 comb；
- 4 个 Tx 的导频并集仍保持约 `1 / pilot_spacing` 的总导频密度。

例如 `pilot_spacing=4, N_tx=4`：

```text
Tx0, Tx1, Tx2, Tx3 分别占用不同 offset 的 1/16 子载波 comb；
四者合起来占用 1/4 RE；
数据 RE 仍约占 3/4。
```

UE 端根据 Tx 专属 pilot mask 分别估计每个 Rx-Tx 对的 CSI，得到：

```text
H_ls[slot, rx, tx, subcarrier, symbol]
```

## 5. 当前 MIMO Equalizer

第一版支持单流传输，不做空间复用。

当前发送策略：

```text
4 个 Tx 发同一条语义/传统符号流，并按 1/sqrt(N_tx) 归一化。
```

等效接收模型：

```text
y_rx = sum_tx H[rx, tx] * x / sqrt(N_tx) + noise
```

UE 侧使用 Rx MMSE combiner 合并 4 根接收天线：

```text
h_eff[rx] = sum_tx H[rx, tx] / sqrt(N_tx)
x_hat = sum_rx conj(h_eff[rx]) * y_rx[rx] / (sum_rx |h_eff[rx]|^2 + noise_var)
```

输出 `x_hat` 是单流符号，继续送入原始 SwinJSCC decoder，因此没有修改 SwinJSCC decoder。

## 6. 复值通信基座模型接口

MIMO CSI 输入会 reshape 为复值多通道：

```text
[B, N_rx, N_tx, N_sc, N_sym]
-> [B, 2 * N_rx * N_tx, N_sc, N_sym]
```

4x4 MIMO 时：

```text
[B, 4, 4, N_sc, N_sym]
-> [B, 32, N_sc, N_sym]
```

模型输出再 reshape 回：

```text
[B, 4, 4, N_sc, N_sym]
```

已做 smoke：

```text
input_channels: [2, 32, 72, 14]
output_channels: [2, 32, 72, 14]
output_mimo: [2, 4, 4, 72, 14]
```

## 7. 最小测试结果

SISO 原路径：

```bash
python run_demo.py \
  --channel-estimator ls \
  --semantic swinjscc \
  --channel rayleigh \
  --snr-db 10 \
  --h264-crf 51 \
  --output-dir outputs/mimo_smoke_siso_ls
```

结果：

| 指标 | 值 |
|---|---:|
| SwinJSCC PSNR | `27.91 dB` |
| Semantic EVM | `-11.35 dB` |
| CE NMSE | `-18.35 dB` |
| `X_tx` shape | `[72, 600, 14]` |
| `Y_rx` shape | `[72, 600, 14]` |
| `H_true` shape | `[72, 600, 14]` |
| `H_est` shape | `[72, 600, 14]` |

4x4 MIMO Rayleigh 10 dB：

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
  --output-dir outputs/mimo_smoke_4x4_ls
```

结果：

| 指标 | 值 |
|---|---:|
| SwinJSCC PSNR | `28.39 dB` |
| Semantic EVM | `-20.64 dB` |
| CE NMSE | `-18.02 dB` |
| `X_tx` shape | `[72, 4, 600, 14]` |
| `Y_rx` shape | `[72, 4, 600, 14]` |
| `H_true` shape | `[72, 4, 4, 600, 14]` |
| `H_est` shape | `[72, 4, 4, 600, 14]` |
| Equalizer | `single_stream_rx_mmse_combiner` |

## 8. 当前限制

这是最小可运行版本，不是最终 MIMO 系统：

- 当前只支持下行 4x4 MIMO，UL CSI feedback 仍使用 SISO 辅助链路；
- 当前是单流 transmit diversity，不是 4 层空间复用；
- ULA 参数已进入配置，但信道暂未加入角度扩展、空间相关矩阵或阵列响应；
- 4x4 MIMO 全 CSI 压缩反馈尚未实现，当前反馈质量只评估代表性 `rx0-tx0` 子信道；
- comm_foundation 的 MIMO 输入接口已支持，但还没有训练 MIMO checkpoint。

下一步如果继续 MIMO，应优先做：

1. MIMO channel spatial correlation / AoA-AoD ULA model；
2. 全 4x4 CSI feedback 压缩；
3. MIMO Dataset-V1；
4. MIMO comm_foundation checkpoint；
5. 空间复用或预编码策略。
