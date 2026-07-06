# voice/ — provenance

- `stt_engine.py`, `vad_capture.py` + `models/silero_vad_v5.onnx`: **ported from sparky**
  (algal/sparky CM4 base, `sparky_mvp/core/`) per the 2026-06-23 STT/TTS backbone decision
  (IMPORT_LEDGER). `stt_engine` = remote_http STT → AI-VM Parakeet :5093; `vad_capture.VAD` =
  Silero v5 ONNX. Self-contained (stdlib/numpy/onnxruntime). Needs `onnxruntime`.
- `stt_frontend.py`, `streaming_handler.py`: NEW SSoT glue — push-driven in-receive() STT front-end
  (VAD endpoint + remote Parakeet) and the AgentVoiceHandler subclass that wires it.
- Future: AEC (sparky `webrtc_aecm` + Debian `libwebrtc-audio-processing`) for open-mic/barge-in (M5);
  s2s `smart_progressive_streaming` partials for lower latency.
