"""Pure unit tests for the VLM `describe` seam (no network).

Covers the stdlib PNG encoder (valid signature/IHDR, decodes back to the same pixels) and the
OpenAI-vision payload shape, with httpx.post monkeypatched. The live endpoint is exercised by
a hardware smoke, not here.
"""

import base64
import struct
import zlib

import numpy as np

from reachy_agent.body import vlm_client


def _rgb_fixture():
    img = np.zeros((6, 9, 3), dtype=np.uint8)
    img[:, 0:3] = (220, 30, 30)
    img[:, 3:6] = (30, 200, 30)
    img[:, 6:9] = (30, 30, 220)
    return img


def _decode_png(data: bytes):
    """Minimal PNG decoder for colour-type-2 8-bit, filter 0 — mirror of the encoder."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, w, h, idat = 8, None, None, b""
    while pos < len(data):
        ln = struct.unpack(">I", data[pos : pos + 4])[0]
        typ = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + ln]
        if typ == b"IHDR":
            w, h, bd, ct = struct.unpack(">IIBB", chunk[:10])
            assert (bd, ct) == (8, 2), "expected 8-bit RGB"
        elif typ == b"IDAT":
            idat += chunk
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * 3
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        row = raw[y * (stride + 1) : (y + 1) * (stride + 1)]
        assert row[0] == 0, "expected filter byte 0"
        out[y] = np.frombuffer(row[1:], dtype=np.uint8).reshape(w, 3)
    return out


def test_png_encoder_roundtrips_pixels():
    img = _rgb_fixture()
    png = vlm_client._png_bytes(img)
    np.testing.assert_array_equal(_decode_png(png), img)


def test_data_url_is_png_base64():
    url = vlm_client._png_data_url(_rgb_fixture())
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_data_url_drops_alpha_and_casts():
    rgba = np.dstack([_rgb_fixture(), np.full((6, 9), 255, np.uint8)])  # HxWx4
    url = vlm_client._png_data_url(rgba.astype(np.float32))  # non-uint8 + alpha
    assert url.startswith("data:image/png;base64,")


def test_describe_payload_shape_and_endpoint(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "  red, green, blue stripes  "}}]}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(vlm_client.httpx, "post", fake_post)
    out = vlm_client.smolvlm_describe(
        _rgb_fixture(),
        "What is visible?",
        base_url="http://vlm.local:3448/v1/",
        model="gemma-4-E4B",
    )
    assert out == "red, green, blue stripes"  # stripped
    assert captured["url"] == "http://vlm.local:3448/v1/chat/completions"  # trailing slash handled
    body = captured["json"]
    assert body["model"] == "gemma-4-E4B"
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "text" and "What is visible?" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_describe_env_defaults(monkeypatch):
    monkeypatch.setenv("AGENT_VLM_BASE_URL", "http://env.local:1234/v1")
    monkeypatch.setenv("AGENT_VLM_MODEL", "env-model")
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json=None, timeout=None):
        captured["url"], captured["model"] = url, json["model"]
        return _Resp()

    monkeypatch.setattr(vlm_client.httpx, "post", fake_post)
    vlm_client.smolvlm_describe(_rgb_fixture(), "q")
    assert captured["url"] == "http://env.local:1234/v1/chat/completions"
    assert captured["model"] == "env-model"


# --- audit 2026-07-02: _png_data_url robustness --------------------------------


def test_png_data_url_grayscale_hw1_expands_to_rgb():
    # (H, W, 1) single-channel must not corrupt the PNG (was written as width-w rows under RGB IHDR).
    gray = np.full((4, 5, 1), 128, dtype=np.uint8)
    url = vlm_client._png_data_url(gray)
    assert url.startswith("data:image/png;base64,")
    pixels = _decode_png(base64.b64decode(url.split(",", 1)[1]))
    assert pixels.shape == (4, 5, 3)
    np.testing.assert_array_equal(pixels[0, 0], (128, 128, 128))  # replicated across channels


def test_png_data_url_float_0_1_frame_is_scaled_not_blackened():
    # A float frame in [0,1] must scale to 0..255, not clip to {0,1} (near-black).
    f = np.ones((3, 3, 3), dtype=np.float32)  # pure white in normalized space
    url = vlm_client._png_data_url(f)
    pixels = _decode_png(base64.b64decode(url.split(",", 1)[1]))
    np.testing.assert_array_equal(pixels[0, 0], (255, 255, 255))  # scaled to white, not (1,1,1)
