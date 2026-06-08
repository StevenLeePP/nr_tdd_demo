# Communication Foundation Model Exploration

## 1. Current NR-TDD Semantic Link

Current end-to-end path:

1. Image input is loaded by `SwinJSCCInterface` in `semantic.py`.
2. SwinJSCC encoder produces semantic complex IQ symbols.
3. `BaseStation.build_downlink()` in `nodes.py` maps semantic symbols and H.264+LDPC symbols onto the DL OFDM resource grid.
4. `ResourceGridMapper` in `resource_grid.py` inserts comb pilots in every OFDM symbol. Pilot:data ratio is 1:3.
5. `OFDMModem` in `ofdm.py` performs IFFT and symbol-dependent normal CP insertion.
6. `MultipathChannel` in `channel.py` applies AWGN/Rayleigh/Rician multipath fading.
7. UE demodulates the received waveform back to `rx_grid`.
8. `ChannelEstimator.estimate_slot()` in `dsp.py` performs LS pilot estimation and 2D interpolation.
9. MMSE equalization uses the estimated channel.
10. Equalized semantic symbols go back to the existing SwinJSCC decoder.
11. CSI is compressed by `compress_csi()` in `dsp.py`.
12. UE maps compressed CSI bits onto UL; BS receives and reconstructs CSI quality via `decompress_csi()` and `csi_nmse()`.

## 2. Existing Signal Tensors

The dataset export sanity check uses `Nsc=72`, `Nslot=1`, `Nsym=14`.

| Tensor | Meaning | Shape in sanity dataset | dtype |
|---|---|---:|---|
| `rx_grid` | Received OFDM resource grid | `[100, 1, 72, 14]` | `complex64` |
| `pilot_obs` | Received pilot REs, zero elsewhere | `[100, 1, 72, 14]` | `complex64` |
| `pilot_mask` | Comb pilot mask | `[100, 1, 72, 14]` | `bool` |
| `H_ls_grid` | LS + interpolation baseline estimate | `[100, 1, 72, 14]` | `complex64` |
| `H_true` | True frequency response from simulated impulse response | `[100, 1, 72, 14]` | `complex64` |
| `equalized_symbols` | Equalized data symbols | `[100, 756]` | `complex64` |
| `semantic_tx_symbols` | Transmitted synthetic payload symbols | `[100, 756]` | `complex64` |

`H_true` is available in simulation because `MultipathChannel.transmit()` returns the sampled time-domain impulse response and, when Doppler is enabled, a per-symbol active-subcarrier `frequency_response_grid`. `impulse_response_to_grid()` is still used for the default block-static case.

Metrics:

- Image PSNR: semantic and conventional branches in `simulation.py`.
- CSI compression and BS recovery NMSE: `_evaluate_csi_feedback()` in `simulation.py`.
- Channel estimation NMSE: `csi_nmse(H_true, H_est)` in `simulation.py`.
- Equalized semantic EVM: `_evm()` in `simulation.py`.
- Training NMSE loss: `nmse_loss()` in `models/comm_foundation_model.py`.
- Learned-estimator inference time: accumulated in `UserEquipment.receive_downlink()` and logged under `channel_estimation_quality`.

## 3. Dataset Export

New script:

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

The script exports `.npz` arrays and a sidecar JSON summary. Channel mode is only a generation control field, not a classification label. Doppler is also a generation/control variable: nonzero Doppler applies lightweight per-path time-varying phases and exports the resulting per-symbol `H_true`. This matches the research direction: the model should serve deployment tasks such as channel estimation and CSI feedback, not identify AWGN/Rayleigh/Rician labels.

Sanity result:

- Samples: `100`
- Mean LS baseline NMSE: `0.0504` in export summary
- Shapes and dtypes verified in `comm_foundation_sanity_100.json`

## 4. Complex Backbone Design

New modules:

- `models/complex_layers.py`
- `models/comm_foundation_model.py`

The current minimal representation uses `[B, 2*C, F, T]`, where the first `C` channels are real and the second `C` channels are imaginary. The layers obey complex multiplication:

```text
Yr = Wr * Xr - Wi * Xi
Yi = Wr * Xi + Wi * Xr
```

Implemented:

- `ComplexConv2d`
- `ComplexLinear`
- `ComplexReLU`
- `ComplexLayerNorm`
- `ComplexCommunicationBackbone`

The output latent is:

```text
Z_comm = ComplexCommunicationBackbone(H_ls_grid)
```

For the default sanity model, `Z_comm` has shape `[B, 2*hidden_complex_channels, F, T]`; with `hidden_complex_channels=16`, this is `[B, 32, F, T]`.

## 5. Reference Paper Design Implications

The PDF `Unveiling the Power of Complex-Valued Transformers in Wireless Communications` motivates the direction beyond this first CNN prototype:

- Wireless signals and channel responses are naturally complex-valued, so phase and amplitude should be modeled natively instead of flattening them into unrelated real channels.
- The paper argues theoretically and experimentally for complex-valued neural networks, and proposes a complex-valued transformer paradigm with embedding, encoding, decoding, and output projection modules.
- The paper studies representative wireless tasks including channel estimation and joint pilot/feedback quantization/precoding design.

Engineering implication for this repo:

- The current `ComplexCommunicationBackbone` is intentionally lightweight for the first closed loop.
- Its interface is designed so it can later be replaced by a Complex Transformer backbone:

```text
Complex embedding over RE/token patches
→ Complex self-attention blocks over time-frequency tokens
→ Z_comm
→ task heads
```

The current heads and training scripts should remain reusable after that replacement.

## 6. Implemented Task Heads

Implemented:

- `ChannelEstimationHead`: predicts `H_hat`.

Reserved:

- `CSIFeedbackHead`: future learned CSI compression/reconstruction.
- `ReliabilityHead`: future time-frequency reliability map.
- `SemanticAssistHead`: future condition feature for SwinJSCC decoder assistance.

These heads live in `comm_foundation_model.py`.

## 7. Training Script

New script:

```bash
python scripts/train_comm_foundation_model.py \
  --dataset_path outputs/comm_foundation_sanity_100.npz \
  --output_dir outputs/comm_foundation_ckpt_sanity \
  --epochs 1 \
  --batch_size 16 \
  --lr 0.001 \
  --use_masked_csi \
  --use_denoising
```

Supported objectives:

```text
L_pretrain = lambda_ce * L_channel_estimation
           + lambda_mask * L_masked_csi
           + lambda_denoise * L_denoising_csi
```

The 1-epoch sanity run produced:

- Validation model NMSE: `0.1789`
- LS baseline NMSE: `0.0451`
- Checkpoint: `outputs/comm_foundation_ckpt_sanity/best_comm_foundation_channel_estimator.pt`

This is not expected to beat LS yet; it verifies the training and checkpoint loop.

## 8. Optional Simulation Integration

The original link remains unchanged by default:

```bash
python run_demo.py --channel-estimator ls
```

The learned estimator path is optional:

```bash
python run_demo.py \
  --channel-estimator comm_foundation \
  --comm-foundation-checkpoint outputs/comm_foundation_ckpt_sanity/best_comm_foundation_channel_estimator.pt
```

When enabled:

1. UE still computes `H_ls_grid` from pilots.
2. `LearnedChannelEstimator` maps `H_ls_grid` to `H_hat`.
3. `H_hat` replaces LS interpolation for equalization.
4. SwinJSCC decoder remains unchanged.
5. Logs include channel-estimation NMSE and semantic EVM.
6. If the learned estimator path is used, logs include model inference time in milliseconds.

## 9. Initial Results

With LS estimator, Rayleigh, 20 dB:

- SwinJSCC PSNR: about `28.10 dB`
- H.264+LDPC PSNR: about `34.16 dB`
- Channel-estimation NMSE: about `-24.17 dB`
- BS recovered CSI NMSE after compressed feedback: about `-5.19 dB`

With the 1-epoch sanity foundation checkpoint:

- SwinJSCC PSNR dropped to about `20.65 dB`
- Channel-estimation NMSE was about `-5.13 dB`

Interpretation: the checkpoint only proves integration. It is not a meaningful pretrained foundation model yet.

## 10. Next Steps

1. Generate a larger dataset across SNR, Doppler, delay profiles, pilot densities, and feedback budgets.
2. Train longer and compare against LS/MMSE baselines on held-out SNR and channel profiles.
3. Replace the lightweight Complex CNN with a PDF-inspired Complex Transformer:
   - complex patch/token embedding for OFDM RE blocks;
   - complex self-attention over time-frequency tokens;
   - complex projection heads for channel estimation and CSI feedback.
4. Add `CSIFeedbackHead` to replace downsampling + scalar quantization.
5. Add `ReliabilityHead` and feed reliability maps to resource allocation or SwinJSCC protection logic.
6. Add semantic-aware losses while keeping SwinJSCC decoder intact.
7. Evaluate generalization under Doppler, sparse pilots, lower SNR, and CSI feedback compression pressure.
