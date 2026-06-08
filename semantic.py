from __future__ import annotations

import contextlib
import hashlib
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

from .config import SemanticConfig
from .utils import ComplexArray, qpsk_modulate


sys.path.append("/root/lap/semantic/SwinJSCC")
try:  # pragma: no cover - depends on the external SwinJSCC checkout.
    import torch
    import torch.nn as nn
    from PIL import Image
    from net.network import SwinJSCC
except Exception:  # noqa: BLE001 - external model imports can fail in many ways.
    torch = None
    nn = None
    Image = None
    SwinJSCC = None


class SwinJSCCInterface:
    """Real SwinJSCC encoder/decoder wrapper plus a deterministic fallback."""

    def __init__(self, cfg: SemanticConfig, snr_db: float, channel_type: str) -> None:
        self.cfg = cfg
        self.snr_db = float(snr_db)
        self.channel_type = channel_type
        self.net = None
        self.device = None
        self.semantic_state: Dict[str, object] = {}
        self.last_reconstruction: Optional[np.ndarray] = None
        self.last_original: Optional[np.ndarray] = None
        self.last_tx_symbols: Optional[np.ndarray] = None
        self.last_metrics: Dict[str, float | int | str] = {}

        if self.cfg.use_real_swinjscc:
            self._load_real_swinjscc()

    def _load_real_swinjscc(self) -> None:
        if torch is None or nn is None or Image is None or SwinJSCC is None:
            raise RuntimeError("PyTorch/PIL/SwinJSCC imports are unavailable.")
        if self.cfg.model_name in {"SwinJSCC_w/_RA", "SwinJSCC_w/_SAandRA"}:
            raise ValueError(
                "This PHY bridge currently targets fixed-rate SwinJSCC models "
                "such as SwinJSCC_w/_SA with C=96."
            )
        if not Path(self.cfg.model_path).exists():
            raise FileNotFoundError(self.cfg.model_path)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args = SimpleNamespace(
            training=False,
            trainset="DIV2K",
            testset="kodak",
            distortion_metric="MSE",
            model=self.cfg.model_name,
            channel_type=self.channel_type,
            C=str(self.cfg.rate),
            multiple_snr=str(self.snr_db),
            model_size=self.cfg.model_size,
        )
        config = self._build_swinjscc_config(args)

        with contextlib.redirect_stdout(io.StringIO()):
            net = SwinJSCC(args, config).to(self.device)
        state_dict = torch.load(self.cfg.model_path, map_location=self.device)
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        self.net = net
        self.args = args
        self.swin_config = config

    def _build_swinjscc_config(self, args: SimpleNamespace) -> SimpleNamespace:
        if self.cfg.model_size == "small":
            encoder_depths = [2, 2, 2, 2]
            decoder_depths = [2, 2, 2, 2]
        elif self.cfg.model_size == "large":
            encoder_depths = [2, 2, 18, 2]
            decoder_depths = [2, 18, 2, 2]
        else:
            encoder_depths = [2, 2, 6, 2]
            decoder_depths = [2, 6, 2, 2]

        image_dims = (3, self.cfg.image_size, self.cfg.image_size)
        encoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=encoder_depths,
            num_heads=[4, 6, 8, 10],
            C=self.cfg.rate,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        decoder_kwargs = dict(
            model=args.model,
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128],
            depths=decoder_depths,
            num_heads=[10, 8, 6, 4],
            C=self.cfg.rate,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )
        return SimpleNamespace(
            seed=42,
            pass_channel=False,
            CUDA=torch.cuda.is_available(),
            device=self.device,
            norm=False,
            logger=None,
            downsample=4,
            image_dims=image_dims,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
        )

    def encode_image(self, image_path: Optional[str], target_symbols: int) -> ComplexArray:
        if self.net is None:
            bits = self._fallback_image_bits(image_path, 2 * target_symbols)
            return qpsk_modulate(bits)[:target_symbols]

        input_tensor, original_np = self._load_image_tensor(image_path)
        with torch.no_grad():
            _, _, height, width = input_tensor.shape
            if height != self.net.H or width != self.net.W:
                self.net.encoder.update_resolution(height, width)
                self.net.decoder.update_resolution(
                    height // (2 ** self.swin_config.downsample),
                    width // (2 ** self.swin_config.downsample),
                )
                self.net.H = height
                self.net.W = width

            feature = self.net.encoder(input_tensor, self.snr_db, self.cfg.rate, self.cfg.model_name)
            if isinstance(feature, tuple):
                feature = feature[0]
            feature = feature.contiguous()
            pwr = torch.mean(feature**2) * 2.0
            channel_tx = feature / torch.sqrt(pwr)

        flat = channel_tx.detach().reshape(-1)
        if flat.numel() % 2:
            flat = torch.cat([flat, flat.new_zeros(1)])
        half = flat.numel() // 2
        symbols = (flat[:half] + 1j * flat[half:]).detach().cpu().numpy().astype(np.complex128)
        if symbols.size > target_symbols:
            raise ValueError(
                f"SwinJSCC latent needs {symbols.size} complex REs, but DL grid only "
                f"has {target_symbols}. Increase n_subcarriers or n_dl_slots."
            )

        self.semantic_state = {
            "feature_shape": tuple(int(x) for x in feature.shape),
            "feature_power": float(pwr.detach().cpu()),
            "n_complex_symbols": int(symbols.size),
            "image_path": image_path,
        }
        self.last_original = original_np
        self.last_reconstruction = None
        self.last_tx_symbols = symbols.copy()
        self.last_metrics = {
            "encoder": "SwinJSCC",
            "model_path": self.cfg.model_path,
            "model_name": self.cfg.model_name,
            "image_size": self.cfg.image_size,
            "n_iq_symbols": int(symbols.size),
            "feature_power": float(pwr.detach().cpu()),
        }
        return symbols

    def decode_symbols(self, symbols: ComplexArray, output_path: Optional[str] = None) -> object:
        if self.net is None:
            return {
                "decoder": "fallback",
                "output_path": output_path,
                "num_iq_symbols": int(symbols.size),
                "avg_symbol_power": float(np.mean(np.abs(symbols) ** 2)) if symbols.size else 0.0,
            }

        n_complex = int(self.semantic_state["n_complex_symbols"])
        symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)[:n_complex]
        flat_real = np.concatenate([np.real(symbols), np.imag(symbols)]).astype(np.float32)

        feature_shape = tuple(int(x) for x in self.semantic_state["feature_shape"])
        feature_power = float(self.semantic_state["feature_power"])
        with torch.no_grad():
            feature_norm = torch.from_numpy(flat_real).to(self.device).reshape(feature_shape)
            recovered_feature = feature_norm * math.sqrt(feature_power)
            recon = self.net.decoder(recovered_feature, self.snr_db, self.cfg.model_name).clamp(0.0, 1.0)

        recon_np = self._tensor_to_image(recon)
        self.last_reconstruction = recon_np
        if output_path is not None:
            self._save_image(recon_np, output_path)

        psnr = self._psnr(self.last_original, recon_np) if self.last_original is not None else float("nan")
        self.last_metrics.update(
            {
                "decoder": "SwinJSCC",
                "output_path": output_path or "",
                "num_iq_symbols": int(n_complex),
                "psnr_db": float(psnr),
            }
        )
        return dict(self.last_metrics)

    def _load_image_tensor(self, image_path: Optional[str]) -> Tuple[object, np.ndarray]:
        if image_path is None:
            raise ValueError("image_path is required for real SwinJSCC encoding.")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        crop = min(width, height, self.cfg.image_size)
        left = (width - crop) // 2
        top = (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop)).resize(
            (self.cfg.image_size, self.cfg.image_size),
            Image.BICUBIC,
        )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, arr

    @staticmethod
    def _tensor_to_image(tensor: object) -> np.ndarray:
        arr = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _save_image(image_np: np.ndarray, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image_u8 = np.rint(np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(image_u8).save(output_path)

    @staticmethod
    def _psnr(reference: Optional[np.ndarray], estimate: np.ndarray) -> float:
        if reference is None:
            return float("nan")
        mse = float(np.mean((np.asarray(reference) - np.asarray(estimate)) ** 2))
        if mse <= 1e-12:
            return float("inf")
        return 10.0 * math.log10(1.0 / mse)

    @staticmethod
    def _fit_length(symbols: ComplexArray, target_symbols: int) -> ComplexArray:
        symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
        if symbols.size >= target_symbols:
            return symbols[:target_symbols]
        return np.pad(symbols, (0, target_symbols - symbols.size), constant_values=0.0)

    @staticmethod
    def _fallback_image_bits(image_path: Optional[str], bit_count: int) -> np.ndarray:
        if image_path is not None and Path(image_path).exists():
            payload = Path(image_path).read_bytes()
        else:
            payload = b"SwinJSCC fallback semantic payload"
        digest = hashlib.sha256(payload).digest()
        repeated = (payload + digest) * max(1, math.ceil(bit_count / (8 * (len(payload) + 32))))
        return np.unpackbits(np.frombuffer(repeated, dtype=np.uint8))[:bit_count]
