# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple
import mitsuba as mi

# -----------------------------------------------------------------------------
# Mitsuba/Dr.Jit setup
# -----------------------------------------------------------------------------
mi.set_variant("cuda_ad_rgb")

from lib.utils.render_utils import (
    PairMetric,
    Results,
    RunConfig,
    make_output_dir_str,
    _ensure_output_dir,
    render_and_time,
    save_np_image,
    save_results,
)
from lib.utils.meta_material import MetaMaterialTypes
import lib.utils.io_utils as io_utils


from lib.utils.render_eval_utils import register_meta_svbrdf

register_meta_svbrdf()

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
    p.add_argument("--spp", type=int, default=4)
    p.add_argument("--metalog-marginal", type=str, default="cos_theta")
    p.add_argument(
        "--meta-material-type-A", type=str, default="meta_roughconductor"
    )
    p.add_argument(
        "--meta-material-type-B", type=str, default="meta_roughconductor"
    )
    p.add_argument("--metalog-dir-A", type=Path, required=True)
    p.add_argument("--metalog-dir-B", type=Path, required=True)
    p.add_argument("--blend-texture-path", type=Path, required=True)
    p.add_argument("--film-width", type=int, default=512)
    p.add_argument("--film-height", type=int, default=512)
    p.add_argument("--benchmark-runs", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--rusinkiewicz-reparam", action="store_true")

    args = p.parse_args()

    # Validate meta material
    try:
        _ = MetaMaterialTypes(args.meta_material_type_A)
    except ValueError:
        raise ValueError(
            f"{args.meta_material_type_A} is not a valid MetaMaterialTypes."
        )
    try:
        _ = MetaMaterialTypes(args.meta_material_type_B)
    except ValueError:
        raise ValueError(
            f"{args.meta_material_type_B} is not a valid MetaMaterialTypes."
        )

    # Resolve IO
    metalog_scene_path = args.metalog_bsdf_scene.resolve()

    # Material A
    metalogs_dir_A = args.metalog_dir_A.resolve()
    io_utils.ensure_dirs(metalogs_dir_A)

    brdf_path_A = metalogs_dir_A / "brdf.exr"
    sample_metalogs_path_A = metalogs_dir_A / "sample_metalogs.pt"
    io_utils.ensure_files(
        brdf_path_A, sample_metalogs_path_A, metalog_scene_path
    )

    # Material B
    metalogs_dir_B = args.metalog_dir_B.resolve()
    io_utils.ensure_dirs(metalogs_dir_B)

    brdf_path_B = metalogs_dir_B / "brdf.exr"
    sample_metalogs_path_B = metalogs_dir_B / "sample_metalogs.pt"
    io_utils.ensure_files(
        brdf_path_B, sample_metalogs_path_B, metalog_scene_path
    )

    blend_texture_path = args.blend_texture_path

    # Output directory
    if args.output_dir is None:
        out_dir = Path(
            make_output_dir_str(f"{OUTPUT_DIR}/metalog_meta_svbsdf_test")
        )
    else:
        out_dir = args.output_dir.resolve()
    _ensure_output_dir(out_dir)

    return args, (
        brdf_path_A,
        brdf_path_B,
        sample_metalogs_path_A,
        sample_metalogs_path_B,
        blend_texture_path,
        out_dir,
    )


def main():
    args, (
        brdf_path_A,
        brdf_path_B,
        sample_metalogs_path_A,
        sample_metalogs_path_B,
        blend_texture_path,
        out_dir,
    ) = parse_args()

    # Load scenes
    metalog_scene = mi.load_file(
        str(args.metalog_bsdf_scene),
        sample_metalogs_filename_0=str(sample_metalogs_path_A),
        sample_metalogs_filename_1=str(sample_metalogs_path_B),
        directional_hemispherical_reflectance_filename_0=str(brdf_path_A),
        directional_hemispherical_reflectance_filename_1=str(brdf_path_B),
        metalog_marginal=args.metalog_marginal,
        meta_material_type_A=args.meta_material_type_A,
        meta_material_type_B=args.meta_material_type_B,
        blend_texture_filename=str(blend_texture_path),
        width=args.film_width,
        height=args.film_height,
        rusinkiewicz_reparam="true" if args.rusinkiewicz_reparam else "false",
    )

    # grab the default sensor
    sensor = metalog_scene.sensors()[0]

    gpu_idx = 0
    img, time = render_and_time(
        metalog_scene,
        args.spp,
        args.benchmark_runs,
        gpu_index=gpu_idx,
        sensor=sensor,
    )

    bmp_path_str = str(out_dir / "meta_svbrdf.png")
    save_np_image(img, bmp_path_str)

    empty_metrics = {
        "MSE": 0.0,
        "FLIP": 0.0,
        "SSIM": 0.0,
    }
    results = Results(
        timings={"I1": time, "I2": time, "I3": time},
        width=args.film_width,
        height=args.film_height,
        metrics={
            "I1_I2": PairMetric("I1_I2", "None", 0.0),
            "I1_I3": PairMetric("I1_I3", "None", 0.0),
            "I2_I3": PairMetric("I2_I3", "None", 0.0),
        },
        metrics_all={
            "I1_I2": empty_metrics,
            "I1_I3": empty_metrics,
            "I2_I3": empty_metrics,
        },
    )

    cfg = RunConfig(
        metalog_bsdf_scene=args.metalog_bsdf_scene,
        reference_bsdf_scene=None,
        spp=args.spp,
        ref_spp_high=args.spp,
        cdf_method="newton",
        metalog_term=-1,
        metalog_marginal=args.metalog_marginal,
        meta_material_type=args.meta_material_type_A,
        metalog_dir=f"A: {str(sample_metalogs_path_A)}\nB: {str(sample_metalogs_path_B)}",
        dr_metalogs=True,
        film_width=int(args.film_width),
        film_height=int(args.film_height),
        measured_material_path=None,
        metric=None,
        benchmark_runs=int(args.benchmark_runs),
        output_dir=out_dir,
        output_format="both",
        save_images=True,
        no_show=False,
        gpu_index=gpu_idx,
        marginal_mode="mixture",
        conditional_mode="mixture",
        mis_active=True,
        rusinkiewicz_reparam=args.rusinkiewicz_reparam,
    )
    save_results(cfg, results, output_path=out_dir)
    print(f"Render complete. Results saved to -> {out_dir}")


if __name__ == "__main__":
    main()
