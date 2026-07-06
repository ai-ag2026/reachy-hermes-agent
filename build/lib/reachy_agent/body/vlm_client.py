"""VLM `describe` seam for one-shot vision — OpenAI-compatible vision chat (Gemma on the AI-VM).

`one_shot_vision(question, capture_frame, describe)` (see vision.py) takes a `describe(frame, question)`
seam. This provides it via the served Gemma vision endpoint (:3448, OpenAI-compatible) — no in-process
model load. Endpoint/model are env/preset-controlled (`$AGENT_VLM_BASE_URL`, `$AGENT_VLM_MODEL`).

The frame (numpy HxWx3 uint8) is encoded to a PNG data-URL and sent as an image_url message. PNG
encoding is pure-stdlib (zlib) so this seam needs only numpy+httpx — no Pillow. The no-raw-image
safety contract (and the opt-in person-ID denylist) is enforced by `one_shot_vision`, not here.
"""

from __future__ import annotations

import base64
import os
import struct
import zlib
from typing import Any

import httpx
import numpy as np

_VISION_HINT = (
    "Describe what is clearly visible — objects, shapes, colours, layout, lighting, and people "
    "(appearance, clothing, posture, expression, and who they might be if recognisable). "
    "Be concrete; if unsure about a detail, say so rather than inventing it."
)


def _png_bytes(arr: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 RGB array to PNG using only the stdlib (zlib)."""
    h, w = arr.shape[:2]
    arr = np.ascontiguousarray(arr)
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))  # filter byte 0 per scanline

    def _chunk(typ: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, colour-type 2 (RGB)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")


def _downscale(arr: np.ndarray, max_dim: int) -> np.ndarray:
    """Anti-aliased (area-average) downscale via numpy block-mean — no Pillow. A full camera frame
    (e.g. 1280x720) makes a large PNG whose encode + base64 + transfer blow the VLM latency budget on
    the robot's wifi; downscaling keeps it fast. Plain striding aliases badly (poor recognition), so
    block-averaging is used to preserve legible detail for the VLM."""
    if max_dim <= 0:
        return arr
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return arr
    step = (m + max_dim - 1) // max_dim
    if step <= 1:
        return arr
    hh, ww = (h // step) * step, (w // step) * step
    if hh == 0 or ww == 0:
        return np.ascontiguousarray(arr[::step, ::step])  # degenerate -> fall back to striding
    block = arr[:hh, :ww].reshape(hh // step, step, ww // step, step, -1).astype(np.float32)
    return block.mean(axis=(1, 3)).clip(0, 255).astype(np.uint8)


def _png_data_url(frame: Any) -> str:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        # Float frames in [0,1] would clip to {0,1} (≈ black) — scale to 0..255 first.
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)  # single-channel grayscale (H,W,1) -> RGB
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]  # drop alpha
    arr = _downscale(arr, int(os.getenv("AGENT_VLM_MAX_DIM", "768")))
    return "data:image/png;base64," + base64.b64encode(_png_bytes(arr)).decode()


def smolvlm_describe(
    frame: Any, question: str, *, base_url: str | None = None, model: str | None = None, timeout: float = 60.0
) -> str:
    """Describe `frame` answering `question` via the served VLM. Returns sanitized text."""
    base_url = (base_url or os.getenv("AGENT_VLM_BASE_URL", "http://127.0.0.1:3448/v1")).rstrip("/")
    model = model or os.getenv("AGENT_VLM_MODEL", "gemma-4-E4B")
    payload = {
        "model": model,
        "max_tokens": 200,
        "temperature": 0.2,
        # Disable chain-of-thought on thinking VLMs (Qwopus-9B) so it describes directly instead of
        # spending max_tokens reasoning; non-thinking servers (SmolVLM2) ignore this kwarg.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{_VISION_HINT}\n\n{question}"},
                    {"type": "image_url", "image_url": {"url": _png_data_url(frame)}},
                ],
            }
        ],
    }
    r = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def describe_frames(
    frames: Any,
    question: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    max_frames: int = 8,
) -> str:
    """Describe a short sequence of frames (a video clip) via the served VLM. Gemma-4-E4B (Gemma 3n)
    handles video; the frames are sent as an ordered set of images in one message. **gemma-only path**
    (defaults to AGENT_VLM_BASE_URL/MODEL, which point at the dedicated gemma vision endpoint)."""
    base_url = (base_url or os.getenv("AGENT_VLM_BASE_URL", "http://127.0.0.1:3448/v1")).rstrip("/")
    model = model or os.getenv("AGENT_VLM_MODEL", "gemma-4-E4B")
    seq = [f for f in (frames or []) if f is not None][: max(1, max_frames)]
    if not seq:
        return ""
    content: list = [
        {
            "type": "text",
            "text": f"{_VISION_HINT}\n\nDies ist eine kurze Bildfolge (Video) in zeitlicher "
            f"Reihenfolge. Beschreibe, was im zeitlichen Verlauf passiert (Bewegung, "
            f"Veränderung). {question}",
        }
    ]
    for fr in seq:
        content.append({"type": "image_url", "image_url": {"url": _png_data_url(fr)}})
    payload = {
        "model": model,
        "max_tokens": 250,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": content}],
    }
    r = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()
