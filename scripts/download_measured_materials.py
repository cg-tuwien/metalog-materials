# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

"""Download the measured materials of Dupuy and Jakob (2018) from the RGL material database.

Uses only the Python standard library, so it can be run on the host without the
project environment (measured_materials/ is mounted read-only in the Docker container).

Files are stored as
    measured_materials/DupuyAndJakob2018/{isotropic,anisotropic}_{rgb,spec}/<name>_{rgb,spec}.bsdf

Examples:
    python scripts/download_measured_materials.py                     # all RGB materials
    python scripts/download_measured_materials.py --format all        # RGB and spectral
    python scripts/download_measured_materials.py --type isotropic
    python scripts/download_measured_materials.py --materials cc_nothern_aurora paper_blue
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://d38rqfq1h7iukm.cloudfront.net/media/materials"
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "measured_materials"
    / "DupuyAndJakob2018"
)

# Material list from https://rgl.epfl.ch/materials
ISOTROPIC = [
    "acrylic_felt_green",
    "acrylic_felt_orange",
    "acrylic_felt_pink",
    "acrylic_felt_purple",
    "acrylic_felt_white",
    "acrylic_felt_yellow",
    "aurora_white",
    "cardboard",
    "cc_amber_citrine",
    "cc_blue_agat",
    "cc_green_malachite",
    "cc_ibiza_sunset",
    "cc_iris_purple_gem",
    "cc_nothern_aurora",
    "cg_sunflower",
    "chm_light_blue",
    "chm_mint",
    "chm_orange",
    "cm_military_green",
    "cm_toxic_green",
    "cm_white",
    "colodur_azure_4e",
    "colodur_connemara_4c",
    "colodur_kalahari_2a",
    "colodur_napoli_4f",
    "ilm_l3_37_dark_green",
    "ilm_l3_37_matte",
    "ilm_l3_37_metallic",
    "ilm_solo_m_68",
    "ilm_solo_millennium_falcon",
    "irid_flake_paint1",
    "irid_flake_paint1_fine",
    "irid_flake_paint2",
    "laika_ceiling_paint_18_gray",
    "leaf_maple",
    "paper_blue",
    "paper_green",
    "paper_red",
    "paper_white",
    "paper_yellow",
    "satin_blue",
    "satin_gold",
    "satin_purple",
    "satin_rosaline",
    "satin_white",
    "spectralon",
    "vch_dragon_eye_red",
    "vch_frozen_amethyst",
    "vch_golden_yellow",
    "vch_silk_blue",
    "vch_ultra_pink",
]

ANISOTROPIC = [
    "aniso_brushed_aluminium_1",
    "aniso_copper_sheet",
    "aniso_green_pvc",
    "aniso_metallic_paper_copper",
    "aniso_metallic_paper_gold",
    "aniso_miro_7",
    "aniso_morpho_melenaus",
    "aniso_sari_silk_2color",
    "ilm_aniso_darth_vader_pants",
    "ilm_aniso_tarkin_tunic",
    "weta_brushed_steel_satin_pink",
]


def download(url: str, dst: Path, force: bool) -> str:
    if dst.is_file() and not force:
        return f"skip      {dst.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(
            tmp, "wb"
        ) as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return f"done      {dst.name}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format",
        choices=["rgb", "spec", "all"],
        default="rgb",
        help="RGB (~100 MB total) and/or spectral (~1.5 GB total) data. Default: rgb",
    )
    parser.add_argument(
        "--type",
        choices=["isotropic", "anisotropic", "all"],
        default="all",
        help="Restrict to isotropic or anisotropic materials. Default: all",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        metavar="NAME",
        help="Only download the given materials (e.g. cc_nothern_aurora)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Parallel downloads. Default: 4"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download existing files"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available materials and exit"
    )
    args = parser.parse_args()

    materials = []
    if args.type in ("isotropic", "all"):
        materials += [(name, "isotropic") for name in ISOTROPIC]
    if args.type in ("anisotropic", "all"):
        materials += [(name, "anisotropic") for name in ANISOTROPIC]

    if args.materials:
        unknown = set(args.materials) - {name for name, _ in materials}
        if unknown:
            sys.exit(
                f"ERROR: unknown material(s) for --type {args.type}: {', '.join(sorted(unknown))}"
            )
        materials = [
            (name, kind) for name, kind in materials if name in args.materials
        ]

    if args.list:
        for name, kind in materials:
            print(f"{kind:12} {name}")
        return

    formats = ["rgb", "spec"] if args.format == "all" else [args.format]
    jobs = [
        (
            f"{BASE_URL}/{name}/{name}_{fmt}.bsdf",
            args.out_dir / f"{kind}_{fmt}" / f"{name}_{fmt}.bsdf",
        )
        for name, kind in materials
        for fmt in formats
    ]

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(download, url, dst, args.force): (url, dst)
            for url, dst in jobs
        }
        for i, future in enumerate(as_completed(futures), 1):
            url, dst = futures[future]
            try:
                msg = future.result()
            except Exception as e:
                failed.append(url)
                msg = f"FAILED    {dst.name}: {e}"
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)

    if failed:
        sys.exit(f"{len(failed)} download(s) failed:\n  " + "\n  ".join(failed))
    print(f"All files available in {args.out_dir}")


if __name__ == "__main__":
    main()
