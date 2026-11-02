# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Run the README pipeline (precomputation + rendering) for example materials.

For each material, this fits a metalog material, renders it next to the
measured reference and reports the FLIP error between both renderings.
The script only fails if a step fails; judging the FLIP values is up to you.
Intended as a quick end-to-end test of an installation, e.g. the Docker
container:

    docker compose run --rm mlm python -m scripts.run_examples

The measured materials and the matpreview scene must be downloaded beforehand
(on the host, as the repository is read-only inside the container):

    python scripts/download_measured_materials.py --materials cc_nothern_aurora vch_golden_yellow
    python scripts/download_matpreview.py

On Windows (Docker Desktop / WSL2), rendering fails with "Could not initialize
OptiX!" unless the OptiX libraries are set up once on the host (Git Bash or
WSL, from the repository root; see README "Windows (Docker Desktop / WSL2)"):

    bash scripts/docker/setup_optix_wsl.sh 590.48.01

and the WSL override that mounts them is included when running the container:

    docker compose -f docker-compose.yml -f docker-compose.wsl.yml run --rm mlm python -m scripts.run_examples

(setup_optix_wsl.sh also writes a .env file with COMPOSE_FILE set accordingly,
so afterwards the plain command above works as well.)

Results are written to output/examples/<timestamp>/<material>/{fit,render}.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MATERIAL_DIR = Path("measured_materials/DupuyAndJakob2018/isotropic_rgb")
SCENE_FILES = [
    Path("scenes/matpreview/matpreview.serialized"),
    Path("scenes/matpreview/envmap.exr"),
]
DEFAULT_MATERIALS = ["cc_nothern_aurora", "vch_golden_yellow"]

# Precomputation parameters from the README
FIT_ARGS = [
    "--material-type", "measured",
    "--material-name", "meta_epfl_measured",
    "--min-regularization-marginal", "0.0001",
    "--max-regularization-marginal", "0.0001",
    "--min-regularization-conditional", "0.0001",
    "--max-regularization-conditional", "0.0001",
    "--min-metalog-resolution-marginal", "16",
    "--max-metalog-resolution-marginal", "16",
    "--min-metalog-resolution-conditional", "16",
    "--max-metalog-resolution-conditional", "16",
    "--n-conditionals", "6",
    "--ml-phi-res", "-1",
    "--ml-theta-res", "-1",
    "--dhr-phi-res", "1",
    "--dhr-theta-res", "16",
    "--metalog-marginal", "phi",
    "--rusinkiewicz-reparam", "true",
    "--no-plots",
    "--marginal-mode", "mixture",
    "--conditional-mode", "mixture",
    "--mix-k-marg", "2",
    "--mix-k-cond", "1",
    "--mix-metalog-term-marg", "6",
    "--mix-metalog-term-cond", "8",
    "--mix-max-iters", "30",
    "--mixture-model", "auto",
]  # fmt: skip

# Rendering parameters from the README
RENDER_ARGS = [
    "--metalog-bsdf-scene", "scenes/matpreview/scene_metalog_meta_mis.xml",
    "--reference-bsdf-scene", "scenes/matpreview/scene_mitsuba_measured_mis.xml",
    "--marginal-mode", "mixture",
    "--conditional-mode", "mixture",
    "--metalog-marginal", "phi",
    "--metric", "FLIP",
    "--save-images",
    "--no-show",
    "--mis-active", "true",
    "--rusinkiewicz-reparam",
    "--meta-material-type", "meta_epfl_measured",
    "--output-format", "json",
]  # fmt: skip


def run_module(module: str, args: list[str]) -> float:
    """Runs `python -m module args`, raises on failure, returns the runtime."""
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def run_material(name: str, out_dir: Path, args) -> dict:
    measured_path = MATERIAL_DIR / f"{name}_rgb.bsdf"
    fit_dir = out_dir / name / "fit"
    render_dir = out_dir / name / "render"

    print(f"\n{'=' * 80}\n{name}: precomputation\n{'=' * 80}", flush=True)
    fit_time = run_module(
        "scripts.fit.run_meta_bsdf_maker",
        [
            *FIT_ARGS,
            "--measured-material-path", str(measured_path),
            "--output-dir", str(fit_dir),
        ],
    )  # fmt: skip

    metalog_dirs = [p.parent for p in fit_dir.rglob("brdf.exr")]
    if len(metalog_dirs) != 1:
        raise RuntimeError(
            f"Expected one brdf.exr in {fit_dir}, found {len(metalog_dirs)}"
        )

    print(f"\n{'=' * 80}\n{name}: rendering\n{'=' * 80}", flush=True)
    render_time = run_module(
        "scripts.render.metalog_meta_bsdf_test",
        [
            *RENDER_ARGS,
            "--spp", str(args.spp),
            "--ref-spp-high", str(args.ref_spp_high),
            "--measured-material-path", str(measured_path),
            "--metalog-dir", str(metalog_dirs[0]),
            "--output-dir", str(render_dir),
        ],
    )  # fmt: skip

    with open(render_dir / "results.json") as f:
        metrics = json.load(f)["results"]["metrics_all"]
    return {
        "material": name,
        # Reference (I2) vs. metalog (I3) rendering at the same spp
        "flip": float(metrics["I2_I3"]["FLIP"]),
        # Reference at high spp (I1) vs. same spp (I2): Monte Carlo noise level
        "flip_noise": float(metrics["I1_I2"]["FLIP"]),
        "fit_time": fit_time,
        "render_time": render_time,
        "render_dir": render_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=DEFAULT_MATERIALS,
        help=f"Isotropic measured materials (default: {' '.join(DEFAULT_MATERIALS)})",
    )
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--ref-spp-high", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/examples")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    args = parser.parse_args()

    missing = [
        str(MATERIAL_DIR / f"{name}_rgb.bsdf")
        for name in args.materials
        if not (MATERIAL_DIR / f"{name}_rgb.bsdf").is_file()
    ]
    if missing:
        sys.exit(
            "ERROR: missing measured material(s):\n  "
            + "\n  ".join(missing)
            + "\nDownload them first (on the host when using Docker):\n"
            + "  python scripts/download_measured_materials.py --materials "
            + " ".join(args.materials)
        )

    missing = [str(path) for path in SCENE_FILES if not path.is_file()]
    if missing:
        sys.exit(
            "ERROR: missing matpreview scene file(s):\n  "
            + "\n  ".join(missing)
            + "\nDownload them first (on the host when using Docker):\n"
            + "  python scripts/download_matpreview.py"
        )

    results = []
    for name in args.materials:
        try:
            results.append(run_material(name, args.output_dir, args))
        except Exception as e:
            print(f"\nERROR: {name} failed: {e}", file=sys.stderr, flush=True)
            results.append({"material": name, "error": str(e)})

    print(f"\n{'=' * 80}\nSummary\n{'=' * 80}")
    print(
        "FLIP: metalog vs. reference rendering (both at --spp); "
        "noise: reference at --spp vs. --ref-spp-high\n"
    )
    for r in results:
        if "error" in r:
            print(f"ERROR  {r['material']:24} {r['error']}")
        else:
            print(
                f"OK     {r['material']:24} FLIP {r['flip']:.4f}  "
                f"(noise {r['flip_noise']:.4f})  fit {r['fit_time']:.0f}s  "
                f"render {r['render_time']:.0f}s  -> {r['render_dir']}"
            )

    sys.exit(1 if any("error" in r for r in results) else 0)


if __name__ == "__main__":
    main()
