# Output

All scripts write their results into this directory. Its contents are not tracked by git; only this file is.

- `bounded/<material-name>[_reparam]_<measured-material>/<metalog-marginal>/<UTC date>/<UTC time>/`: Precomputation results (`brdf.exr`, `sample_metalogs.pt`, logs and optional plots).
- `render/metalog_meta_bsdf_test/<UTC date>/<UTC time>/`: Rendering results (comparison plots, metrics and optional images).
- `examples/<timestamp>/<material>/`: Results of the example run (`scripts/run_examples.py`).

See the [main README](../README.md) for details.
