# test-assets

Generated fixtures, not committed (see `.gitignore`). Rebuild with:

    python3 tests/natron/make_test_assets.py test-assets

Produces `plate_srgb8.png`, `plate_acescg.exr`, `trimap_int8.png`, `trimap_float.exr` and
`alpha_gt.png`. The plate and the trimap are each written in two encodings so the four
combinations can be cross-checked — they must all yield the same matte.
