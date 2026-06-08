from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .config import ConventionalConfig


try:  # pragma: no cover - depends on system FFmpeg codecs through PyAV.
    import av
except Exception:  # noqa: BLE001
    av = None


@dataclass
class ConventionalTxPacket:
    tx_bits: np.ndarray
    ldpc_bits: np.ndarray
    original_bits: np.ndarray
    original_image: np.ndarray
    source_h264_reconstruction: np.ndarray
    metadata: Dict[str, int | float | str]


class SimpleLDPCCodec:
    """Small systematic sparse parity code with syndrome bit-flip decoding.

    H = [A | I], codeword = [data, A data]. It is intentionally lightweight and
    swappable; the H264 wrapping is independent from this LDPC fallback.
    """

    def __init__(
        self,
        data_bits: int = 512,
        parity_bits: int = 256,
        row_weight: int = 6,
        max_iters: int = 25,
        seed: int = 2026,
    ) -> None:
        if data_bits <= 0 or parity_bits <= 0:
            raise ValueError("LDPC data_bits and parity_bits must be positive.")
        if row_weight <= 0 or row_weight > data_bits:
            raise ValueError("LDPC row_weight must be in [1, data_bits].")
        self.k = int(data_bits)
        self.m = int(parity_bits)
        self.n = self.k + self.m
        self.max_iters = int(max_iters)
        rng = np.random.default_rng(seed)
        self.rows = np.vstack(
            [rng.choice(self.k, size=row_weight, replace=False) for _ in range(self.m)]
        ).astype(np.int32)

    def encode(self, bits: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
        pad_bits = (-bits.size) % self.k
        if pad_bits:
            bits = np.pad(bits, (0, pad_bits), constant_values=0)
        blocks = bits.reshape(-1, self.k)
        encoded_blocks = []
        for block in blocks:
            parity = np.bitwise_xor.reduce(block[self.rows], axis=1).astype(np.uint8)
            encoded_blocks.append(np.concatenate([block, parity]))
        return np.concatenate(encoded_blocks), {
            "ldpc_data_bits": self.k,
            "ldpc_parity_bits": self.m,
            "ldpc_pad_bits": int(pad_bits),
            "ldpc_blocks": int(blocks.shape[0]),
        }

    def decode(self, received_bits: np.ndarray, original_bit_count: int) -> Tuple[np.ndarray, Dict[str, int]]:
        received_bits = np.asarray(received_bits, dtype=np.uint8).reshape(-1)
        usable = (received_bits.size // self.n) * self.n
        received_bits = received_bits[:usable]
        if usable == 0:
            return np.zeros(original_bit_count, dtype=np.uint8), {
                "ldpc_iterations": 0,
                "ldpc_final_syndrome_weight": original_bit_count,
            }

        blocks = received_bits.reshape(-1, self.n).copy()
        total_iters = 0
        final_syndrome_weight = 0
        decoded_blocks = []
        for block in blocks:
            data = block[: self.k].copy()
            parity = block[self.k :].copy()
            syndrome = self._syndrome(data, parity)
            for iteration in range(self.max_iters):
                if not np.any(syndrome):
                    break
                scores = np.zeros(self.k, dtype=np.int16)
                bad_rows = np.flatnonzero(syndrome)
                for row_idx in bad_rows:
                    scores[self.rows[row_idx]] += 1
                flip_idx = int(np.argmax(scores))
                if scores[flip_idx] >= 2:
                    data[flip_idx] ^= 1
                else:
                    # With H=[A|I], an isolated unsatisfied check is most
                    # likely a parity-bit error; flip that parity bit directly.
                    parity[int(bad_rows[0])] ^= 1
                syndrome = self._syndrome(data, parity)
                total_iters += 1
            final_syndrome_weight += int(np.sum(syndrome))
            decoded_blocks.append(data)

        decoded = np.concatenate(decoded_blocks)[:original_bit_count]
        return decoded.astype(np.uint8), {
            "ldpc_iterations": int(total_iters),
            "ldpc_final_syndrome_weight": int(final_syndrome_weight),
        }

    def _syndrome(self, data: np.ndarray, parity: np.ndarray) -> np.ndarray:
        expected = np.bitwise_xor.reduce(data[self.rows], axis=1).astype(np.uint8)
        return expected ^ parity


class H264LDPCImageCodec:
    """Traditional baseline: H.264 image compression plus LDPC channel coding."""

    def __init__(self, cfg: ConventionalConfig) -> None:
        self.cfg = cfg
        self.ldpc = SimpleLDPCCodec(
            data_bits=cfg.ldpc_data_bits,
            parity_bits=cfg.ldpc_parity_bits,
            row_weight=cfg.ldpc_row_weight,
            max_iters=cfg.ldpc_max_iters,
        )

    def encode_image(self, image_path: str) -> ConventionalTxPacket:
        image = self._load_image(image_path)
        h264_bytes = self._encode_h264(image)
        source_recon = self._decode_h264(h264_bytes)
        payload_bits = np.unpackbits(np.frombuffer(h264_bytes, dtype=np.uint8))
        ldpc_bits, ldpc_meta = self.ldpc.encode(payload_bits)
        # Interleaved repetition: send full LDPC codeword copies back-to-back so
        # repetitions experience separated time-frequency resources.
        tx_bits = np.tile(ldpc_bits, self.cfg.repetition_factor).astype(np.uint8)
        metadata: Dict[str, int | float | str] = {
            **ldpc_meta,
            "codec": "H264+LDPC",
            "h264_bytes": int(len(h264_bytes)),
            "payload_bits": int(payload_bits.size),
            "ldpc_coded_bits": int(ldpc_bits.size),
            "transmitted_bits": int(tx_bits.size),
            "h264_crf": int(self.cfg.h264_crf),
            "image_size": int(self.cfg.image_size),
            "ldpc_rate": float(self.ldpc.k / self.ldpc.n),
            "repetition_factor": int(self.cfg.repetition_factor),
        }
        return ConventionalTxPacket(
            tx_bits=tx_bits,
            ldpc_bits=ldpc_bits,
            original_bits=payload_bits,
            original_image=image,
            source_h264_reconstruction=source_recon,
            metadata=metadata,
        )

    def decode_bits(
        self,
        received_bits: np.ndarray,
        original_bit_count: int,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, int | float | str]]:
        ldpc_input_bits = self._majority_combine_repetitions(received_bits)
        decoded_bits, ldpc_meta = self.ldpc.decode(ldpc_input_bits, original_bit_count)
        byte_count = int(math.ceil(original_bit_count / 8))
        decoded_bytes = np.packbits(decoded_bits)[:byte_count].tobytes()
        metadata: Dict[str, int | float | str] = {
            **ldpc_meta,
            "decoded_payload_bits": int(decoded_bits.size),
            "decoded_h264_bytes": int(byte_count),
            "received_ldpc_bits_after_repetition": int(ldpc_input_bits.size),
        }
        try:
            image = self._decode_h264(decoded_bytes)
            metadata["h264_decode_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            image = np.zeros((self.cfg.image_size, self.cfg.image_size, 3), dtype=np.float32)
            metadata["h264_decode_status"] = f"failed: {type(exc).__name__}"
        if output_path is not None:
            self.save_image(image, output_path)
        return image, metadata

    def _majority_combine_repetitions(self, received_bits: np.ndarray) -> np.ndarray:
        received_bits = np.asarray(received_bits, dtype=np.uint8).reshape(-1)
        rep = int(self.cfg.repetition_factor)
        if rep <= 1:
            return received_bits
        copy_len = received_bits.size // rep
        grouped = received_bits[: copy_len * rep].reshape(rep, copy_len)
        return (np.sum(grouped, axis=0) >= ((rep // 2) + 1)).astype(np.uint8)

    def _load_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        crop = min(width, height, self.cfg.image_size)
        left = (width - crop) // 2
        top = (height - crop) // 2
        image = image.crop((left, top, left + crop, top + crop)).resize(
            (self.cfg.image_size, self.cfg.image_size),
            Image.BICUBIC,
        )
        return np.asarray(image, dtype=np.float32) / 255.0

    def _encode_h264(self, image: np.ndarray) -> bytes:
        if av is None:
            raise RuntimeError("PyAV is required for H.264 encoding.")
        image_u8 = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=1)
        stream.width = self.cfg.image_size
        stream.height = self.cfg.image_size
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "crf": str(self.cfg.h264_crf),
            "preset": "veryfast",
            "tune": "stillimage",
        }
        frame = av.VideoFrame.from_ndarray(image_u8, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return buffer.getvalue()

    def _decode_h264(self, payload: bytes) -> np.ndarray:
        if av is None:
            raise RuntimeError("PyAV is required for H.264 decoding.")
        container = av.open(io.BytesIO(payload), mode="r", format="mp4")
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
            return np.clip(arr, 0.0, 1.0)
        raise ValueError("No frame decoded from H.264 payload.")

    @staticmethod
    def save_image(image_np: np.ndarray, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image_u8 = np.rint(np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(image_u8).save(output_path)
