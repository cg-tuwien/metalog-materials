# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

"""
Metalog Meta-BSDF benchmark & comparison tool
  - Render three images: I1 (reference, high-spp), I2 (reference, user-spp), I3 (metalog, user-spp)
  - Benchmark render times of I1/I2/I3 using Dr.Jit kernel history (see benchmark_utils)
  - Compute perceptual error metrics between all pairs: (I1,I2), (I1,I3), (I2,I3)
      - Metric is SSIM by default ("SSIM"), optional "FLIP" using NVLabs FLIP
  - Save configuration & results to JSON/CSV
  - Plot images + error maps with concise annotations
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Tuple, Any
import mitsuba as mi

# -----------------------------------------------------------------------------
# Mitsuba/Dr.Jit setup
# -----------------------------------------------------------------------------
mi.set_variant("cuda_ad_rgb")

from lib.utils.render_utils import (
    RunConfig,
    make_output_dir_str,
    _ensure_output_dir,
    full_render_comparison,
)
from lib.utils.meta_material import MetaMaterialTypes
from lib.bsdfs import metalog_meta_bsdf
import lib.utils.io_utils as io_utils


# Register MetaBSDF
mi.register_bsdf("MetaBSDF", lambda props: metalog_meta_bsdf.MetaBSDF(props))

import drjit as dr

DRJIT_SYMBOLIC_MODE = True
dr.set_flag(dr.JitFlag.SymbolicCalls, DRJIT_SYMBOLIC_MODE)
dr.set_flag(dr.JitFlag.LoopRecord, DRJIT_SYMBOLIC_MODE)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
OUTPUT_DIR = "output/render"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> Tuple[RunConfig, Path]:
    p = argparse.ArgumentParser()
    p.add_argument("--metalog-bsdf-scene", type=Path, required=True)
    p.add_argument("--reference-bsdf-scene", type=Path, required=True)
    p.add_argument("--spp", type=int, default=4)
    p.add_argument(
        "--ref-spp-high", type=int, default=4096, help="I1 spp (default 4096)"
    )
    p.add_argument("--cdf-method", type=str, default="newton")
    p.add_argument("--metalog-term", type=int, default=5)
    p.add_argument("--metalog-marginal", type=str, default="phi")
    p.add_argument(
        "--meta-material-type", type=str, default="meta_roughconductor"
    )
    p.add_argument("--metalog-dir", type=Path, required=True)
    p.add_argument("--no-dr-metalogs", dest="dr_metalogs", action="store_false")
    p.add_argument("--film-width", type=int, default=512)
    p.add_argument("--film-height", type=int, default=512)
    p.add_argument("--measured-material-path", type=Path, default=None)
    p.add_argument(
        "--metric",
        type=str,
        choices=["SSIM", "FLIP"],
        default="FLIP",
        help="Error metric (default: FLIP)",
    )
    p.add_argument("--benchmark-runs", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--output-format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
    )
    p.add_argument(
        "--marginal-mode",
        type=str,
        choices=["metalog", "mixture"],
        default="metalog",
    )
    p.add_argument(
        "--conditional-mode",
        type=str,
        choices=["metalog", "mixture"],
        default="metalog",
    )
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index for NVML memory sampling (default: 0)",
    )
    p.add_argument(
        "--mis-active",
        type=str,
        choices=["true", "false"],
        default="true",
    )
    p.add_argument("--rusinkiewicz-reparam", action="store_true")

    args = p.parse_args()

    # Validate meta material
    try:
        _ = MetaMaterialTypes(args.meta_material_type)
    except ValueError:
        raise ValueError(
            f"{args.meta_material_type} is not a valid MetaMaterialTypes."
        )

    # Resolve IO
    ref_scene_path = args.reference_bsdf_scene.resolve()
    metalog_scene_path = args.metalog_bsdf_scene.resolve()

    metalogs_dir = args.metalog_dir.resolve()
    io_utils.ensure_dirs(metalogs_dir)

    brdf_path = metalogs_dir / "brdf.exr"
    sample_metalogs_path = metalogs_dir / "sample_metalogs.pt"
    io_utils.ensure_files(
        brdf_path, sample_metalogs_path, ref_scene_path, metalog_scene_path
    )

    measured_path = args.measured_material_path
    if args.meta_material_type == "meta_epfl_measured":
        if measured_path is not None and str(measured_path) != "None":
            measured_path = measured_path.resolve()
            io_utils.ensure_files(measured_path)

    # Output directory
    if args.output_dir is None:
        out_dir = Path(
            make_output_dir_str(f"{OUTPUT_DIR}/metalog_meta_bsdf_test")
        )
    else:
        out_dir = args.output_dir.resolve()
    _ensure_output_dir(out_dir)

    cfg = RunConfig(
        metalog_bsdf_scene=metalog_scene_path,
        reference_bsdf_scene=ref_scene_path,
        spp=int(args.spp),
        ref_spp_high=int(args.ref_spp_high),
        cdf_method=args.cdf_method,
        metalog_term=int(args.metalog_term),
        metalog_marginal=args.metalog_marginal,
        meta_material_type=args.meta_material_type,
        metalog_dir=metalogs_dir,
        dr_metalogs=bool(args.dr_metalogs),
        film_width=int(args.film_width),
        film_height=int(args.film_height),
        measured_material_path=str(measured_path) if measured_path else None,
        metric=args.metric,
        benchmark_runs=int(args.benchmark_runs),
        output_dir=out_dir,
        output_format=args.output_format,
        save_images=bool(args.save_images),
        no_show=bool(args.no_show),
        gpu_index=int(args.gpu_index),
        marginal_mode=args.marginal_mode,
        conditional_mode=args.conditional_mode,
        mis_active=args.mis_active,
        rusinkiewicz_reparam=args.rusinkiewicz_reparam,
    )
    return cfg, (brdf_path, sample_metalogs_path, measured_path)


def load_scenes(
    cfg: RunConfig, brdf_path: Path, sample_metalogs_path: Path
) -> Tuple[Any, Any]:
    # Metalog scene (I3)
    metalog_scene = mi.load_file(
        str(cfg.metalog_bsdf_scene),
        sample_metalogs_filename=str(sample_metalogs_path),
        directional_hemispherical_reflectance_filename=str(brdf_path),
        metalog_terms=cfg.metalog_term,
        metalog_marginal=cfg.metalog_marginal,
        meta_material_type=cfg.meta_material_type,
        cdf_method=cfg.cdf_method,
        dr_metalogs=cfg.dr_metalogs,
        width=cfg.film_width,
        height=cfg.film_height,
        marginal_mode=cfg.marginal_mode,
        conditional_mode=cfg.conditional_mode,
        rusinkiewicz_reparam=cfg.rusinkiewicz_reparam,
    )

    # Reference scene (I1/I2)
    ref_kwargs = dict(width=cfg.film_width, height=cfg.film_height)
    if (
        cfg.meta_material_type == "meta_epfl_measured"
        and cfg.measured_material_path
    ):
        ref_scene = mi.load_file(
            str(cfg.reference_bsdf_scene),
            material_filename=str(cfg.measured_material_path),
            **ref_kwargs,
        )
    else:
        ref_scene = mi.load_file(str(cfg.reference_bsdf_scene), **ref_kwargs)

    if cfg.mis_active == "true":
        # check if paths contain the "mis" extension
        if "mis" not in str(cfg.metalog_bsdf_scene):
            logging.warning(
                f"`cfg.mis_active` is set to true, but file Metalog scene file name extension does not have 'mis'."
            )
        if "mis" not in str(cfg.reference_bsdf_scene):
            logging.warning(
                f"`cfg.mis_active` is set to true, but file reference scene file name extension does not have 'mis'."
            )

    return metalog_scene, ref_scene


def main():
    cfg, (brdf_path, sample_metalogs_path, _measured) = parse_args()

    # Load scenes
    metalog_scene, ref_scene = load_scenes(cfg, brdf_path, sample_metalogs_path)

    # grab the default sensor
    sensors = [metalog_scene.sensors()[0]]

    full_render_comparison(
        ref_scene,
        metalog_scene,
        sensors,
        cfg=cfg,
    )
    print(f"Render complete. Results saved to -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
