# Headless host tests

| File | Purpose |
|---|---|
| `check_plugin.py` | Plugin discovery + node creation in `NatronRenderer`. Also prints the input labels, so a missing Trimap clip shows up. Prints `RESULT: PASS`/`FAIL`. |
| `render_matte.py` | Read(plate) + Read(trimap) → MEMatte → Write. Configured by environment variables (`HYP_*`); sets the reader/writer OCIO spaces to identity so the matte is treated as data. |
| `make_test_assets.py` | Generates the fixture set: the same scene as 8-bit sRGB and linear ACEScg EXR, and the same trimap as 8-bit 0/128/255 and float 0/0.5/1. |

The fixtures exist for the colour/trimap matrix check: every combination of plate encoding and
trimap encoding must produce the same matte. See `docs/LINUX.md` for the full run-through.
