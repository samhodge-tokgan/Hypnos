# tools

| File | Purpose |
|---|---|
| `mematte_routing.py` | The export-friendly rewrite of MEMatte's token routing, plus a transcription of upstream's eval path to check it against. **Read its module docstring first** — it is the design document for the export. |
| `export_mematte.py` | Exports the backbone and decoder ONNX graphs. **Not yet run** (needs MEMatte checkpoints + detectron2). |
| `routing_equivalence_test.py` | Proves the rewrite matches upstream and exports with a runtime token cap. No checkpoints needed. |
| `shape_dynamics_test.py` | Proves one graph serves every resolution. No checkpoints needed. |
| `fetch_models.sh` / `.ps1` | Pull published ONNX models into a bundle's `Contents/Resources`. Checksums are pinned once a `models-v1` release exists. |
| `requirements.txt` | Python deps for the above (detectron2 is separate — see the file). |

Both test scripts run in CI and are the gate on the export design.
