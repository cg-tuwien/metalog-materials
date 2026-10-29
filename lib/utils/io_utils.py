# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import logging
import sys
from pathlib import Path
import torch
import lib.metalogs as ml


def ensure_dirs(*dirs: Path):
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        sys.exit(
            "ERROR: missing required directory(ies):\n  " + "\n  ".join(missing)
        )


def ensure_files(*files: Path):
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        sys.exit("ERROR: missing required file(s):\n  " + "\n  ".join(missing))


def setup_logging(log_dir: Path):
    """Configures logging to file and console with different levels."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the lowest level for the logger

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Create file handler for INFO level and above
    file_handler = logging.FileHandler(log_dir / "execution.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create console handler for WARNING level and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def log_marginal_metalog_infeasability(
    max_num_terms: int,
    log_info: str,
    marginal_metalogs: torch.Tensor,
):
    for term in range(3, max_num_terms):
        coeffs = ml.get_metalog_coefficients(marginal_metalogs, term)
        is_marginal_metalog_feasible = ml.is_feasible_metalog(coeffs)
        if not is_marginal_metalog_feasible:
            logging.warning(
                f"({log_info}/wo): {term}-Term Marginal metalog is not feasible."
            )


def log_conditional_metalog_infeasibility(
    metalog_res_marg: int,
    max_num_terms: int,
    log_info: str,
    conditional_metalogs: torch.Tensor,
):
    for res in range(metalog_res_marg):
        cond_coeffs = conditional_metalogs[res]
        for term in range(3, max_num_terms):
            coeffs = ml.get_metalog_coefficients(cond_coeffs, term)
            is_cond_metalog_feasible = ml.is_feasible_metalog(coeffs)
            if not is_cond_metalog_feasible:
                logging.warning(
                    f"({log_info}/wo) Slice {res}, {term}-Term Conditional metalog is not feasible."
                )


def export_mix_metalog_state(
    scene,
    out_dir_str: str,
    original_metalogs_path: str | Path,
    marg_mix_ml_coeffs_key: str,
    cond_mix_ml_coeffs_key: str,
    marg_quantiles_key: str,
    directional_reflectance_key: str,
    base_color_key: str,
    marg_mix_weights_key: str | None = None,
    cond_mix_weights_key: str | None = None,
    dir_postfix: str = "",
):
    import torch
    import mitsuba as mi

    # Base output directory
    base_output_dir = Path(out_dir_str)
    if dir_postfix:
        base_output_dir = base_output_dir / dir_postfix
    base_output_dir.mkdir(parents=True, exist_ok=True)

    params = mi.traverse(scene)

    # reconstruct sample_metalogs.pt using the original structure
    if original_metalogs_path and Path(original_metalogs_path).exists():
        # load the original template mapping
        pt_dict = torch.load(original_metalogs_path, map_location="cpu")
        keys_list = list(pt_dict.keys())
        n_slices = len(keys_list)
        first_entry = pt_dict[keys_list[0]]

        def get_tensor(key):
            """Helper to extract and return CPU PyTorch tensor from DrJit params"""
            return params[key].torch().cpu() if key in params else None

        # fetch the flattened optimized parameters
        marg_coeffs_flat = get_tensor(marg_mix_ml_coeffs_key)
        cond_coeffs_flat = get_tensor(cond_mix_ml_coeffs_key)
        marg_quantiles_flat = get_tensor(marg_quantiles_key)
        marg_weights_flat = get_tensor(marg_mix_weights_key)
        cond_weights_flat = get_tensor(cond_mix_weights_key)

        # reshape flat arrays back into ND shapes based on the template
        if marg_coeffs_flat is not None:
            shape = (n_slices, *first_entry["marginal_mixture"]["coeffs"].shape)
            marg_coeffs_nd = marg_coeffs_flat.view(shape)

        if cond_coeffs_flat is not None:
            if "conditional_mixture" in first_entry:
                shape = (
                    n_slices,
                    *first_entry["conditional_mixture"]["coeffs"].shape,
                )
            else:
                shape = (n_slices, *first_entry["conditional"].shape)
            cond_coeffs_nd = cond_coeffs_flat.view(shape)

        if marg_quantiles_flat is not None:
            shape = (n_slices, *first_entry["marginal_inverse_cdf"].shape)
            marg_quantiles_nd = marg_quantiles_flat.view(shape)

        if marg_weights_flat is not None:
            shape = (
                n_slices,
                *first_entry["marginal_mixture"]["mixing_coeffs"].shape,
            )
            marg_weights_nd = marg_weights_flat.view(shape)

        if cond_weights_flat is not None:
            shape = (
                n_slices,
                *first_entry["conditional_mixture"]["mixing_coeffs"].shape,
            )
            cond_weights_nd = cond_weights_flat.view(shape)

        # inject the reshaped optimized parameters back into the dictionary
        for i, key in enumerate(keys_list):
            if marg_coeffs_flat is not None:
                pt_dict[key]["marginal_mixture"]["coeffs"] = marg_coeffs_nd[i]
            if marg_quantiles_flat is not None:
                pt_dict[key]["marginal_inverse_cdf"] = marg_quantiles_nd[i]
            if marg_weights_flat is not None:
                pt_dict[key]["marginal_mixture"]["mixing_coeffs"] = (
                    marg_weights_nd[i]
                )

            if cond_coeffs_flat is not None:
                if "conditional_mixture" in pt_dict[key]:
                    pt_dict[key]["conditional_mixture"]["coeffs"] = (
                        cond_coeffs_nd[i]
                    )
                else:
                    pt_dict[key]["conditional"] = cond_coeffs_nd[i]
            if cond_weights_flat is not None:
                pt_dict[key]["conditional_mixture"]["mixing_coeffs"] = (
                    cond_weights_nd[i]
                )

        # Save the fully compliant dictionary
        torch.save(pt_dict, base_output_dir / "sample_metalogs.pt")

    # save Directional Hemispherical Reflectance
    if directional_reflectance_key in params:
        dhr_tensor = params[directional_reflectance_key]
        mi.Bitmap(dhr_tensor).write(str(base_output_dir / "dhr.exr"))

    # save base color
    if base_color_key in params:
        base_color = params[base_color_key]
        mi.Bitmap(base_color).write(str(base_output_dir / "base_color.exr"))
