# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import flip_evaluator as nv_flip
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as skimage_ssim
from lib.utils import benchmark_utils as bench
from lib.utils.RotationConfig import RotationConfig
from lib.utils.memory_utils import render_with_mem
import mitsuba as mi
import numpy as np
from typing import Dict, Tuple
import drjit as dr
import logging


@dataclass
class RenderStats:
    mean: float
    median: float
    std: float
    count: int
    cpu_rss_peak_mb: float | None = None
    gpu_vram_peak_mb: float | None = None


@dataclass
class PairMetric:
    pair: str
    metric_name: str
    score: float


@dataclass
class Results:
    # Times are seconds (from bench.benchmark_mitsuba_scene_render)
    timings: Dict[str, RenderStats]
    # Metric scores for each pair (chosen/default metric)
    metrics: Dict[str, PairMetric]
    # All additional scalar metrics per pair (e.g., {"I1_I2":{"MSE":..., "FLIP":...}})
    metrics_all: Dict[str, Dict[str, float]]
    # Basic image and meta info
    width: int
    height: int


@dataclass
class RunConfig:
    metalog_bsdf_scene: Path
    reference_bsdf_scene: Path
    spp: int
    ref_spp_high: int
    cdf_method: str
    metalog_term: int
    metalog_marginal: str
    meta_material_type: str
    metalog_dir: Path
    dr_metalogs: bool
    marginal_mode: str | None
    conditional_mode: str | None
    film_width: int
    film_height: int
    measured_material_path: str | None
    metric: str  # "SSIM" or "FLIP"
    benchmark_runs: int
    output_dir: Path
    output_format: str  # json/csv/both
    save_images: bool
    no_show: bool
    gpu_index: int
    mis_active: bool
    rusinkiewicz_reparam: bool | None


def mi_to_srgb01(img, as_hdr: bool = False) -> np.ndarray:
    """Convert Mitsuba image. Returns raw linear HDR if as_hdr=True, else applies 8-bit sRGB."""
    if as_hdr:
        return np.array(mi.Bitmap(img), dtype=np.float32)

    bmp = mi.util.convert_to_bitmap(img, uint8_srgb=True)
    arr_uint8 = np.array(bmp, dtype=np.uint8)
    arr = (arr_uint8 / np.iinfo(arr_uint8.dtype).max).astype(np.float32)
    return arr


def render_and_time(
    scene,
    spp: int,
    runs: int,
    *,
    gpu_index: int,
    sensor: mi.Sensor | None = None,
    seed: int = 0,
    as_hdr: bool = False,
) -> Tuple[np.ndarray, RenderStats]:
    # Render once (to get the image) while sampling memory
    if sensor:
        img, rss_mb, vram_mb = render_with_mem(
            lambda: mi.render(scene, spp=spp, sensor=sensor, seed=seed),
            gpu_index=gpu_index,
        )
    else:
        img, rss_mb, vram_mb = render_with_mem(
            lambda: mi.render(scene, spp=spp, seed=seed),
            gpu_index=gpu_index,
        )

    # DrJit fence: ensure no state bleeds into Metalog timings
    dr.flush_malloc_cache()
    dr.kernel_history_clear()

    # Benchmark using DrJit kernel history
    stats = bench.benchmark_mitsuba_scene_render(
        scene, spp=spp, n_runs=runs, seed=seed
    )

    dr.flush_malloc_cache()
    dr.kernel_history_clear()

    rstats = RenderStats(
        **stats,
        cpu_rss_peak_mb=float(rss_mb) if rss_mb is not None else None,
        gpu_vram_peak_mb=float(vram_mb) if vram_mb is not None else None,
    )

    return mi_to_srgb01(img, as_hdr=as_hdr), rstats


def flip_map(
    imgA: np.ndarray, imgB: np.ndarray, mode: str = "LDR"
) -> Tuple[np.ndarray, float]:
    """
    FLIP via NVLabs' evaluator.

    Parameters
    ----------
    imgA, imgB : np.ndarray
        HxWx3, float in [0,1]. Our pipeline produces gamma-encoded sRGB, so LDR is default.
    mode : "LDR" or "HDR"
        Use "LDR" for gamma-encoded sRGB inputs, or "HDR" for linear HDR EXR images.

    Returns
    -------
    flip_map_2d : np.ndarray (H x W, float in [0,1])
        Per-pixel FLIP error map.
    mean_flip : float
        Mean FLIP error.
    """
    if mode == "HDR":
        ref = imgA.astype(np.float32)
        test = imgB.astype(np.float32)
    else:
        ref = np.clip(imgA, 0.0, 1.0).astype(np.float32)
        test = np.clip(imgB, 0.0, 1.0).astype(np.float32)

    flipErrorMap, meanFLIPError, _ = nv_flip.evaluate(ref, test, mode)
    fmap = np.asarray(flipErrorMap, dtype=np.float32)

    # if fmap has 3 channels, then average across these channels
    if fmap.ndim == 3 and fmap.shape[2] > 1:
        fmap = fmap.mean(axis=2)

    return np.clip(fmap, 0.0, 1.0), float(meanFLIPError)


def compute_metric_map(
    cfg: RunConfig,
    A: np.ndarray,
    B: np.ndarray,
    is_hdr: bool = False,
) -> Tuple[np.ndarray, float]:
    if cfg.metric == "SSIM":
        # Dynamic range calculation for HDR
        d_range = (
            float(np.maximum(A.max(), B.max()) - np.minimum(A.min(), B.min()))
            if is_hdr
            else 1.0
        )
        if d_range == 0:
            d_range = 1.0

        # skimage expects data_range; images are in [0,1]
        score, s_map = skimage_ssim(
            A,
            B,
            channel_axis=-1,
            data_range=d_range,
            gaussian_weights=True,
            full=True,
        )
        s_map = s_map.astype(np.float32)
        s_map = np.clip(s_map, -1.0, 1.0)
        dssim_map = (1.0 - s_map) * 0.5
        return dssim_map, float(score)
    elif cfg.metric == "FLIP":
        e_map, e = flip_map(A, B, mode="HDR" if is_hdr else "LDR")
        return e_map, e
    else:
        raise ValueError("Unknown metric")


def save_np_image(img01: np.ndarray, path: Path, is_hdr: bool = False) -> None:
    from PIL import Image

    # Apply tonemap strictly for saving the PNG if it's HDR
    if is_hdr:
        img01 = np.clip(img01, 0.0, 1.0)
        img01 = np.where(
            img01 <= 0.0031308,
            12.92 * img01,
            1.055 * np.power(img01, 1.0 / 2.4) - 0.055,
        )

    arr8 = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def compute_additional_metrics(
    A: np.ndarray, B: np.ndarray, is_hdr: bool = False
) -> Dict[str, float]:
    """
    Extra scalar metrics shown next to the chosen default metric.
    """
    out: Dict[str, float] = {}
    out["MSE"] = float(np.mean((A - B) ** 2))

    # Handle HDR vs LDR for FLIP
    mode = "HDR" if is_hdr else "LDR"
    ref = (
        A.astype(np.float32) if is_hdr else np.clip(A, 0, 1).astype(np.float32)
    )
    test = (
        B.astype(np.float32) if is_hdr else np.clip(B, 0, 1).astype(np.float32)
    )

    _, meanFLIPError, _ = nv_flip.evaluate(ref, test, mode)
    out["FLIP"] = float(meanFLIPError)

    # Handle HDR vs LDR for SSIM
    d_range = (
        float(np.maximum(A.max(), B.max()) - np.minimum(A.min(), B.min()))
        if is_hdr
        else 1.0
    )
    if d_range == 0:
        d_range = 1.0

    mssim = skimage_ssim(
        A,
        B,
        channel_axis=-1,
        data_range=d_range,
        gaussian_weights=True,
    )
    out["SSIM"] = float(mssim)

    return out


def make_output_dir_str(dir_ext: str) -> str:
    utc_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    utc_time = datetime.now(tz=timezone.utc).time().strftime("%H-%M-%S")
    return f"{dir_ext}/{utc_date}/{utc_time}"


def _ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_triplet_with_errors(
    I1: np.ndarray,
    I2: np.ndarray,
    I3: np.ndarray,
    E12: np.ndarray,
    E13: np.ndarray,
    E23: np.ndarray,
    s12: float,
    s13: float,
    s23: float,
    timings: Dict[str, RenderStats],
    cfg: RunConfig,
    save_path: Path | None = None,
    # Extra scalar metrics to show in titles
    metrics12: Dict[str, float] | None = None,
    metrics13: Dict[str, float] | None = None,
    metrics23: Dict[str, float] | None = None,
    cap12_main_title: str = "Reference",
    cap3_main_title: str = "Ours",
    cap_metric: str = "Error",
    is_hdr: bool = False,
):
    # mimic LaTex text rendering
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.formatter.use_mathtext": True,
        }
    )

    """Grid: top row images (I1,I2,I3), bottom row error maps (E12,E13,E23)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metric_name = cfg.metric

    def _to_scalar01(err):
        err = np.asarray(err)
        if err.ndim == 3:
            # Convert RGB(A) to luminance
            err = (
                0.2126 * err[..., 0]
                + 0.7152 * err[..., 1]
                + 0.0722 * err[..., 2]
            )
        # clip/normalize if needed
        err = np.clip(err, 0.0, 1.0)
        return err

    def _show_img(ax, img, title):
        if is_hdr:
            img = np.clip(img, 0.0, 1.0)
            img = np.where(
                img <= 0.0031308,
                12.92 * img,
                1.055 * np.power(img, 1.0 / 2.4) - 0.055,
            )
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def _show_err(ax, err, title):
        err_scalar = _to_scalar01(err)
        im = ax.imshow(err_scalar, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    def _fmt(
        pair_label: str,
        default_name: str,
        default_score: float,
        extras: Dict[str, float] | None,
    ) -> str:
        # Example: "SSIM(I1,I2)=0.9903, FLIP(I1,I2)=0.0215, MSE(I1,I2)=0.000120"
        txt = f"{default_name}({pair_label})={default_score:.4f}"
        if extras is not None:
            if "SSIM" in extras and metric_name != "SSIM":
                txt += f", SSIM({pair_label})={extras['SSIM']:.4f}"
            if "FLIP" in extras and metric_name != "FLIP":
                txt += f", FLIP({pair_label})={extras['FLIP']:.4f}"
            if "MSE" in extras:
                txt += f", MSE({pair_label})={extras['MSE']:.6f}"
        return txt

    # Titles w/ timings (use each image's timing)
    t1 = timings["I1"]
    t2 = timings["I2"]
    t3 = timings["I3"]

    cap1 = (
        f"spp={cfg.ref_spp_high}, time={t1.mean:.3f}s, runs={cfg.benchmark_runs}"
        if t1.count > 0
        else ""
    )
    cap2 = (
        f"spp={cfg.spp}, time={t2.mean:.3f}s, runs={cfg.benchmark_runs}"
        if t2.count > 0
        else ""
    )
    cap3 = f"spp={cfg.spp}, time={t3.mean:.3f}s, runs={cfg.benchmark_runs}"

    _show_img(axes[0, 0], I1, f"I1: {cap12_main_title}\n {cap1}")
    _show_img(
        axes[0, 1], I2, f"I2: {cap12_main_title} (spp={cfg.spp})\n {cap2}"
    )
    _show_img(
        axes[0, 2],
        I3,
        f"I3: {cap3_main_title}\n {cap3}",
    )

    cap_err1 = _fmt("I1,I2", metric_name, s12, metrics12)
    cap_err2 = _fmt("I1,I3", metric_name, s13, metrics13)
    cap_err3 = _fmt("I2,I3", metric_name, s23, metrics23)

    _show_err(axes[1, 0], E12, f"{cap_metric}\n {cap_err1}")
    _show_err(axes[1, 1], E13, f"{cap_metric}\n {cap_err2}")
    _show_err(axes[1, 2], E23, f"{cap_metric}\n {cap_err3}")

    # Compact run config note
    note = (
        f"metalog-terms={cfg.metalog_term}, marginal={cfg.metalog_marginal}, "
        f"mis-active={cfg.mis_active}, material={cfg.meta_material_type}, size={cfg.film_width}x{cfg.film_height}"
    )
    fig.text(0.5, 0, note, ha="center", va="bottom", fontsize=9)

    # Leave room at bottom for the footer text
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if not cfg.no_show:
        plt.show()
    else:
        plt.close(fig)


def plot_compact_comparison(
    I2: np.ndarray,
    I3: np.ndarray,
    E23: np.ndarray,
    s23: float,
    timings: Dict[str, RenderStats],
    cfg: RunConfig,
    save_path: Path | None = None,
    metrics23: Dict[str, float] | None = None,
    cap1_main_title: str = "Reference",
    cap2_main_title: str = "Ours",
    cap3_main_title: str = "Error",
    is_hdr: bool = False,
):
    # mimic LaTex text rendering
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.formatter.use_mathtext": True,
        }
    )

    """Grid: 1 row, 3 columns (Reference, Ours, Error Map)"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_name = cfg.metric

    def _to_scalar01(err):
        err = np.asarray(err)
        if err.ndim == 3:
            err = (
                0.2126 * err[..., 0]
                + 0.7152 * err[..., 1]
                + 0.0722 * err[..., 2]
            )
        return np.clip(err, 0.0, 1.0)

    def _show_img(ax, img, title):
        if is_hdr:
            img = np.clip(img, 0.0, 1.0)
            img = np.where(
                img <= 0.0031308,
                12.92 * img,
                1.055 * np.power(img, 1.0 / 2.4) - 0.055,
            )
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def _show_err(ax, err, title):
        err_scalar = _to_scalar01(err)
        im = ax.imshow(err_scalar, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    def _fmt(
        default_name: str,
        default_score: float,
        extras: Dict[str, float] | None,
    ) -> str:
        txt = f"{default_name}={default_score:.4f}"
        if extras is not None:
            if "SSIM" in extras and metric_name != "SSIM":
                txt += f", SSIM={extras['SSIM']:.4f}"
            if "FLIP" in extras and metric_name != "FLIP":
                txt += f", FLIP={extras['FLIP']:.4f}"
            if "MSE" in extras:
                txt += f", MSE={extras['MSE']:.6f}"
        return txt

    t2 = timings["I2"]
    t3 = timings["I3"]

    cap1 = (
        f"spp={cfg.spp}, time={t2.mean:.3f}s, runs={cfg.benchmark_runs}"
        if t2.count > 0
        else ""
    )
    cap2 = f"spp={cfg.spp}, time={t3.mean:.3f}s, runs={cfg.benchmark_runs}"
    cap3 = _fmt(metric_name, s23, metrics23)

    _show_img(axes[0], I2, f"{cap1_main_title}\n {cap1}")
    _show_img(axes[1], I3, f"{cap2_main_title}\n {cap2}")
    _show_err(axes[2], E23, f"{cap3_main_title}\n {cap3}")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if not cfg.no_show:
        plt.show()
    else:
        plt.close(fig)


def save_results(cfg: RunConfig, res: Results, output_path: Path) -> None:
    # Convert dataclasses to JSON-friendly dicts (paths->str)
    def _json_ready(o):
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, dict):
            return {k: _json_ready(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_json_ready(v) for v in o]
        return o

    # --- JSON payload pieces ---
    metrics_out = {
        k: {"pair": v.pair, "metric": v.metric_name, "score": v.score}
        for k, v in res.metrics.items()
    }

    # Keep timings pure (no memory fields)
    timings_out = {
        k: {
            "mean": v.mean,
            "median": v.median,
            "std": v.std,
            "count": v.count,
        }
        for k, v in res.timings.items()
    }

    # memory section (pulled from RenderStats if present)
    memory_out = {
        k: {
            "cpu_rss_peak_mb": getattr(v, "cpu_rss_peak_mb", None),
            "gpu_vram_peak_mb": getattr(v, "gpu_vram_peak_mb", None),
        }
        for k, v in res.timings.items()
    }

    payload = {
        "config": _json_ready(asdict(cfg)),
        "results": {
            "width": res.width,
            "height": res.height,
            "metric_name": (
                next(iter(res.metrics.values())).metric_name
                if res.metrics
                else cfg.metric
            ),
            "metrics": metrics_out,
            "metrics_all": res.metrics_all,
            "timings": timings_out,
            "memory": memory_out,
        },
    }

    # --- results.json ---
    if cfg.output_format in ("json", "both"):
        jpath = output_path / "results.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logging.info(f"Saved JSON -> {jpath}")

    # --- CSVs ---
    if cfg.output_format in ("csv", "both"):
        import csv

        # metrics.csv (pair,metric,score)
        mpath = output_path / "metrics.csv"
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair", "metric", "score"])
            for k in ["I1_I2", "I1_I3", "I2_I3"]:
                pm = res.metrics[k]
                w.writerow([pm.pair, pm.metric_name, f"{pm.score:.6f}"])
        logging.info(f"Saved CSV (metrics) -> {mpath}")

        # timings.csv (image,mean,median,std,count)
        tpath = output_path / "timings.csv"
        with open(tpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "mean_s", "median_s", "std_s", "count"])
            for imgk in ["I1", "I2", "I3"]:
                rs = res.timings[imgk]
                w.writerow([imgk, rs.mean, rs.median, rs.std, rs.count])
        logging.info(f"Saved CSV (timings) -> {tpath}")

        # memory.csv (image,cpu_rss_peak_mb,gpu_vram_peak_mb)
        mempath = output_path / "memory.csv"
        with open(mempath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "cpu_rss_peak_mb", "gpu_vram_peak_mb"])
            for imgk in ["I1", "I2", "I3"]:
                rs = res.timings[imgk]
                cpu = getattr(rs, "cpu_rss_peak_mb", None)
                gpu = getattr(rs, "gpu_vram_peak_mb", None)
                w.writerow(
                    [
                        imgk,
                        "" if cpu is None else cpu,
                        "" if gpu is None else gpu,
                    ]
                )
        logging.info(f"Saved CSV (memory) -> {mempath}")

        # long-form CSV with all extra metrics
        amap = output_path / "metrics_all.csv"
        with open(amap, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair", "metric", "score"])
            for pair_key, md in res.metrics_all.items():
                for mname, mval in md.items():
                    w.writerow([pair_key, mname, f"{mval:.6f}"])
        logging.info(f"Saved CSV (metrics_all) -> {amap}")


def full_render_comparison(
    ref: mi.Scene | list,
    ml_scene: mi.Scene,
    sensors: list,
    cfg: RunConfig,
    plotname_postfix: str = "",
    ml_rotate_config: RotationConfig = None,
    ref_rotate_config: RotationConfig = None,
    cap12_main_title: str = "Reference",
    cap3_main_title: str = "Ours",
    cap_metric: str = "Error",
    is_hdr: bool = False,
):
    """
    Makes full rendering comparison across all provided views,
    then saves the error image and metrics.

    `ref` can be a `mi.Scene` or a list of reference images. If latter, then `len(sensors)` must match `len(ref)`.
    """

    # Base output directory
    base_output_dir = Path(cfg.output_dir)
    if plotname_postfix:
        base_output_dir = base_output_dir / plotname_postfix

    n_sensors = len(sensors)
    is_multi_view = n_sensors > 1

    # check whether `ref` is a scene or a list of images
    is_ref_scene = True
    if isinstance(ref, list):
        is_ref_scene = False
        n_images = len(ref)

        # ensure identical number of views
        if n_sensors != n_images:
            raise ValueError("'len(sensors)' do not match 'len(ref)'.")

    ref_rotate_shape = ref_rotate_config is not None and is_ref_scene

    ml_params = mi.traverse(ml_scene)
    ref_params = None
    if is_ref_scene:
        ref_params = mi.traverse(ref)

    for view_idx, sensor in enumerate(sensors):
        # Create a sub-directory for each view if multi-view is active
        current_out_dir = (
            base_output_dir / f"view_{view_idx}"
            if is_multi_view
            else base_output_dir
        )
        os.makedirs(current_out_dir, exist_ok=True)

        # if available, apply rotations to the object
        if ml_rotate_config is not None:
            ml_params[ml_rotate_config.transform_key] = make_mesh_transform(
                ml_rotate_config.mesh_center,
                ml_rotate_config.rot_angles[view_idx],
                ml_rotate_config.up_axis,
            )
            ml_params.update()
        if ref_rotate_shape:
            ref_params[ref_rotate_config.transform_key] = make_mesh_transform(
                ref_rotate_config.mesh_center,
                ref_rotate_config.rot_angles[view_idx],
                ref_rotate_config.up_axis,
            )
            ref_params.update()

        # Pass the specific sensor into the render_and_time function
        if is_ref_scene:
            I1, t1 = render_and_time(
                ref,
                cfg.ref_spp_high,
                cfg.benchmark_runs,
                gpu_index=cfg.gpu_index,
                sensor=sensor,
                as_hdr=is_hdr,
            )
            I2, t2 = render_and_time(
                ref,
                cfg.spp,
                cfg.benchmark_runs,
                gpu_index=cfg.gpu_index,
                sensor=sensor,
                as_hdr=is_hdr,
            )
        else:
            # ref is a list of images
            target_img = ref[view_idx]
            I1 = mi_to_srgb01(target_img, as_hdr=is_hdr)
            I2 = I1.copy()

            # generate dummy timings
            t1 = RenderStats(mean=0.0, median=0.0, std=0.0, count=0)
            t2 = RenderStats(mean=0.0, median=0.0, std=0.0, count=0)

        # Meta BRDF is always a scene
        I3, t3 = render_and_time(
            ml_scene,
            cfg.spp,
            cfg.benchmark_runs,
            gpu_index=cfg.gpu_index,
            sensor=sensor,
            as_hdr=is_hdr,
        )

        # Metrics between pairs
        E12, s12 = compute_metric_map(cfg, I1, I2, is_hdr=is_hdr)
        E13, s13 = compute_metric_map(cfg, I1, I3, is_hdr=is_hdr)
        E23, s23 = compute_metric_map(cfg, I2, I3, is_hdr=is_hdr)

        # Additional scalar metrics
        metrics12 = compute_additional_metrics(I1, I2, is_hdr=is_hdr)
        metrics13 = compute_additional_metrics(I1, I3, is_hdr=is_hdr)
        metrics23 = compute_additional_metrics(I2, I3, is_hdr=is_hdr)

        # Plot & optionally save images
        plot_compact_comparison(
            I2,
            I3,
            E23,
            s23,
            timings={"I2": t2, "I3": t3},
            cfg=cfg,
            save_path=current_out_dir / f"comparison_compact.svg",
            metrics23=metrics23,
            cap1_main_title=cap12_main_title,
            cap2_main_title=cap3_main_title,
            cap3_main_title=cap_metric,
            is_hdr=is_hdr,
        )
        plot_triplet_with_errors(
            I1,
            I2,
            I3,
            E12,
            E13,
            E23,
            s12,
            s13,
            s23,
            timings={"I1": t1, "I2": t2, "I3": t3},
            cfg=cfg,
            save_path=current_out_dir / f"comparison_full.svg",
            metrics12=metrics12,
            metrics13=metrics13,
            metrics23=metrics23,
            cap12_main_title=cap12_main_title,
            cap3_main_title=cap3_main_title,
            cap_metric=cap_metric,
            is_hdr=is_hdr,
        )

        if cfg.save_images:
            # always save the core images used in compact mode
            save_np_image(I2, current_out_dir / "I2_ref_spp.png", is_hdr=is_hdr)
            save_np_image(I3, current_out_dir / "I3_metalog.png", is_hdr=is_hdr)
            save_np_image(
                E23, current_out_dir / f"E23_{cfg.metric}.png", is_hdr=is_hdr
            )

            # Only save the high-spp ref and cross-metrics if we have a true
            # reference scene and aren't forcing a strict compact output
            if is_ref_scene:
                save_np_image(
                    I1, current_out_dir / "I1_ref_high.png", is_hdr=is_hdr
                )
                save_np_image(
                    E12,
                    current_out_dir / f"E12_{cfg.metric}.png",
                    is_hdr=is_hdr,
                )
                save_np_image(
                    E13,
                    current_out_dir / f"E13_{cfg.metric}.png",
                    is_hdr=is_hdr,
                )

        logging.info(
            f"Saved error images for view {view_idx} to -> {current_out_dir}"
        )

        # Save structured results
        # Note: We keep all keys intact so downstream CSV parsers don't break.
        # If is_ref_scene=False, I1_I2 metrics will simply reflect a perfect match (0 error).
        results = Results(
            timings={"I1": t1, "I2": t2, "I3": t3},
            metrics={
                "I1_I2": PairMetric("I1_I2", cfg.metric, s12),
                "I1_I3": PairMetric("I1_I3", cfg.metric, s13),
                "I2_I3": PairMetric("I2_I3", cfg.metric, s23),
            },
            metrics_all={
                "I1_I2": metrics12,
                "I1_I3": metrics13,
                "I2_I3": metrics23,
            },
            width=cfg.film_width,
            height=cfg.film_height,
        )
        save_results(cfg, results, output_path=base_output_dir)


def make_mesh_transform(
    center: mi.Vector3f, angle_deg: float, up_axis: list = [0, 0, 1]
) -> mi.Transform4f:
    """
    Create the 'to_world' rotation transform the given vertices around their bounding box center along the specified
    up-axis.
    """
    if angle_deg == 0.0:
        return mi.Transform4f()

    # translate to origin, rotate and translate back
    t_to_origin = mi.Transform4f.translate(-center)
    rot = mi.Transform4f.rotate(axis=up_axis, angle=angle_deg)
    t_to_center = mi.Transform4f.translate(center)
    transform = t_to_center @ rot @ t_to_origin

    return transform
