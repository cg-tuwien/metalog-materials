# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from lib.utils import io_utils
import lib.metalogs as ml
import lib.utils.plot_metalogs as pm
from lib.fit.meta_bsdf_maker import MetaBSDFMaker
from lib.utils.io_utils import setup_logging
from lib.utils.meta_material import MetaMaterialTypes


def generate_plots(
    args,
    meta_material,
    plots_dir,
    logger,
    sample_metalogs_dict,
):
    logger.info("Generating plots...")

    for slice_key in sample_metalogs_dict.keys():
        slice_data = sample_metalogs_dict[slice_key]
        marginal_bounds, cond_bounds = meta_material.get_bounds()

        plot_save_dir = plots_dir / slice_key
        os.makedirs(plot_save_dir, exist_ok=True)

        if args.conditional_mode == "metalog":
            cond_quantiles = slice_data["conditional_inverse_cdf"]
            conditional_metalogs = slice_data["conditional"]
            fig_cond = pm.plot_metalog_coefficients(
                ml.get_metalog_coefficients(
                    conditional_metalogs, metalog_term=6
                ),
                metalog_inverse_cdf=cond_quantiles,
                bounds=cond_bounds,
                fig_suptitle=f"Conditional Metalogs: {slice_key}",
            )
            save_path_cond = plot_save_dir / "conditional_metalogs.png"
            fig_cond.savefig(save_path_cond)
            plt.close(fig_cond)

        if args.marginal_mode == "metalog":
            orig_samples = slice_data.get("data")
            orig_samples_x = (
                orig_samples[:, 0] if orig_samples is not None else None
            )
            marginal_metalogs = slice_data["marginal"]
            marginal_score = slice_data["marginal_score"]
            fig_marg = pm.plot_metalogs(
                slice_data["empirical_marginal_quantiles"],
                args.metalog_resolution_marginal,
                f"Marginal Metalogs: {slice_key}",
                x_lim=[
                    marginal_bounds[0] - 0.1,
                    marginal_bounds[1] + 0.1,
                ],
                metalog_coefficients=marginal_metalogs,
                num_terms=torch.arange(3, args.max_num_terms),
                scores=marginal_score,
                bounds=marginal_bounds,
                orig_samples=orig_samples_x,
            )
            save_path_marg = plot_save_dir / "marginal_metalogs.png"
            fig_marg.savefig(save_path_marg)
            plt.close(fig_marg)

    logger.info(f"Plots saved in: {plots_dir / 'plots'}")


def build_metalog_parameters(args):
    return {
        "n_samples": args.n_samples,
        "min_regularization_term_marginal": args.min_regularization_marginal,
        "max_regularization_term_marginal": args.max_regularization_marginal,
        "min_regularization_term_conditional": args.min_regularization_conditional,
        "max_regularization_term_conditional": args.max_regularization_conditional,
        "min_metalog_res_marg": args.min_metalog_resolution_marginal,
        "max_metalog_res_marg": args.max_metalog_resolution_marginal,
        "min_metalog_res_cond": args.min_metalog_resolution_conditional,
        "max_metalog_res_cond": args.max_metalog_resolution_conditional,
        "max_num_terms": args.max_num_terms,
        "metalog_marginal": args.metalog_marginal,
        "is_bounded": True,
        "reparam": args.rusinkiewicz_reparam,
        "marginal_mode": args.marginal_mode,
        "conditional_mode": args.conditional_mode,
        "mix_k_marg": args.mix_k_marg,
        "mix_k_cond": args.mix_k_cond,
        "mix_alpha": args.mix_alpha,
        "mix_max_iters": args.mix_max_iters,
        "mix_metalog_term_marg": args.mix_metalog_term_marg,
        "mix_metalog_term_cond": args.mix_metalog_term_cond,
        "mix_model": args.mixture_model,
        "n_conditionals": args.n_conditionals,
    }


def init_args():
    """Initializes and parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Create and save Metalog BSDF distributions for a given material."
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base directory to save the generated metalog files and plots (default: output)",
    )
    p.add_argument(
        "--material-type",
        type=str,
        default="roughconductor",
        help="The type of Mitsuba BSDF to model (default: roughconductor)",
    )
    p.add_argument(
        "--material-name",
        type=str,
        default=MetaMaterialTypes.META_ROUGHCONDUCTOR.value,
        help="The name for the output metalog material (default: meta_roughconductor)",
    )
    p.add_argument(
        "--measured-material-path",
        type=Path,
        default="None",
        help="The path of the EPFL measured material. (default: 'None')",
    )
    p.add_argument(
        "--min-regularization-marginal",
        type=float,
        default=0.005,
        help="Minimum regularization term for marginal metalogs (default: 0.005).",
    )
    p.add_argument(
        "--max-regularization-marginal",
        type=float,
        default=0.005,
        help="Maximum regularization term for marginal metalogs (default: 0.005).",
    )
    p.add_argument(
        "--min-regularization-conditional",
        type=float,
        default=0.005,
        help="Minimum regularization term for condtional metalogs (default: 0.005).",
    )
    p.add_argument(
        "--max-regularization-conditional",
        type=float,
        default=0.005,
        help="Maximum regularization term for condtional metalogs (default: 0.005).",
    )
    p.add_argument(
        "--alpha-u",
        type=float,
        default=0.3,
        help="Roughness value for the u-axis of the BSDF (default: 0.3)",
    )
    p.add_argument(
        "--alpha-v",
        type=float,
        default=0.3,
        help="Roughness value for the v-axis of the BSDF (default: 0.3)",
    )
    p.add_argument(
        "--max-num-terms",
        type=int,
        default=12,
        help="Maximum number of terms to evaluate for metalog fitting (default: 12)",
    )
    p.add_argument(
        "--min-metalog-resolution-marginal",
        type=int,
        default=6,
        help="Minimum resolution for generating marginal metalogs (default: 6)",
    )
    p.add_argument(
        "--max-metalog-resolution-marginal",
        type=int,
        default=32,
        help="Maximum resolution for generating marginal metalogs (default: 32)",
    )
    p.add_argument(
        "--min-metalog-resolution-conditional",
        type=int,
        default=6,
        help="Minimum resolution for generating conditional metalogs (default: 6)",
    )
    p.add_argument(
        "--max-metalog-resolution-conditional",
        type=int,
        default=6,
        help="Maximum resolution for generating conditional metalogs (default: 32)",
    )
    p.add_argument(
        "--n-conditionals",
        type=int,
        default=6,
        help="Number of conditionals (default: 6)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=2**19,
        help="Number of samples to generate for fitting the metalog distributions (default: 2**19)",
    )
    p.add_argument(
        "--mix-metalog-term-marg",
        type=int,
        default=5,
        help="Metalog term for the marginal metalog mixture model (default: 5)",
    )
    p.add_argument(
        "--mix-metalog-term-cond",
        type=int,
        default=5,
        help="Metalog term for the conditional metalog mixture model (default: 5)",
    )
    p.add_argument(
        "--metalog-marginal",
        type=str,
        default="phi",
        choices=["phi", "cos_theta"],
        help="The marginal variable for the bivariate metalog: azimuth (phi) or ~elevation (cos_theta) (default: phi)",
    )
    p.add_argument(
        "--rusinkiewicz-reparam",
        type=str,
        default="phi",
        choices=["true", "false"],
        help="Use Rusinkiewicz reparameterization (default: false)",
    )
    p.add_argument(
        "--make-probs-method",
        type=str,
        default="classic",
        help="CDF probability method (default: 'classic', else: 'chebyshev', 'linspace')",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable the generation and display of plots",
    )
    p.add_argument(
        "--marginal-mode",
        choices=["metalog", "mixture"],
        default="metalog",
        help="Marginal modeling strategy: 'metalog' (legacy n-term) or 'mixture' (SPTMetalogMixtureModel).",
    )
    p.add_argument(
        "--conditional-mode",
        choices=["metalog", "mixture"],
        default="metalog",
        help="Conditional modeling strategy: 'metalog' (legacy n-term) or 'mixture' (MetalogMixtureModel|SPTMetalogMixtureModel).",
    )
    p.add_argument(
        "--mix-k-marg",
        type=int,
        default=2,
        help="Number of components K for the marginal mixture (default: 2)",
    )
    p.add_argument(
        "--mix-k-cond",
        type=int,
        default=2,
        help="Number of components K for the conditional mixture (default: 2)",
    )
    p.add_argument(
        "--mix-alpha",
        type=float,
        default=0.3,
        help="SPT alpha for (q_alpha, q_0.5, q_{1-alpha}) in the marginal SPT mixture (default: 0.3)",
    )
    p.add_argument(
        "--mix-max-iters",
        type=int,
        default=30,
        help="Max EM iterations for the marginal SPT mixture (default: 30)",
    )
    p.add_argument(
        "--ml-theta-res",
        type=int,
        default=-1,
        help="Incident elevation angle resolution for metalogs (default: -1, negative means use default)",
    )
    p.add_argument(
        "--ml-phi-res",
        type=int,
        default=-1,
        help="Incident elevation angle resolution for metalogs (default: -1, negative means use default)",
    )
    p.add_argument(
        "--dhr-theta-res",
        type=int,
        default=16,
        help="Incident elevation angle resolution for directional albedo (default: -1, negative means use default)",
    )
    p.add_argument(
        "--dhr-phi-res",
        type=int,
        default=32,
        help="Incident elevation angle resolution for directional albedo (default: -1, negative means use default)",
    )
    p.add_argument(
        "--mixture-model",
        choices=["auto", "mmm"],
        default="mmm",
        help="Mixture model strategy: 'auto' (AutoMetalogMixture1D) or 'mmm' (MetalogMixtureModel1D).",
    )
    p.add_argument("--anisotropic", action="store_true")
    p.add_argument("--additional-theta-i", action="store_true")

    args = p.parse_args()

    if args.material_type == "measured":
        if (
            args.measured_material_path != None
            and args.measured_material_path != "None"
        ):
            args.measured_material_path = args.measured_material_path.resolve()
            io_utils.ensure_files(
                args.measured_material_path,
            )
        else:
            raise ValueError(
                "Please provide a measured material name from the EPFL material database."
            )
    else:
        args.measured_material_path = None

    return args


def main():
    """Main function to run the Metalog BSDF creation process."""
    args = init_args()

    # --- Initialize Metalog Material Maker to get the output directory ---
    # This allows us to set up logging for the specific output folder from the start.
    args.use_nd_metalogs = True
    args.rusinkiewicz_reparam = args.rusinkiewicz_reparam == "true"
    if args.material_type == "roughconductor":
        mi_bsdf_dict = {
            "type": args.material_type,
            "material": "Cu",
            "distribution": "ggx",
            "alpha_u": args.alpha_u,
            "alpha_v": args.alpha_v,
            "sample_visible": True,
        }
    elif args.material_type == "diffuse":
        args.use_nd_metalogs = False
        mi_bsdf_dict = {
            "type": args.material_type,
            "reflectance": {"type": "rgb", "value": [0.2, 0.25, 0.7]},
        }
    elif args.material_type == "measured":
        mi_bsdf_dict = {
            "type": args.material_type,
            "filename": str(args.measured_material_path),
        }
    else:
        logging.critical(f"Unsupported material type: {args.material_type}")
        raise ValueError(f"Unsupported material type: {args.material_type}")

    is_bounded = True
    resources_dir = str(args.output_dir)
    meta_material = MetaBSDFMaker(
        mi_bsdf_dict,
        metalog_marginal=args.metalog_marginal,
        is_bounded=is_bounded,
        resources_dir=(
            str(args.output_dir) if not args.no_plots else None
        ),  # MetaBSDFMaker will put slice plots into resources_dir
        marginal_mode=args.marginal_mode,
        conditional_mode=args.conditional_mode,
        reparam=args.rusinkiewicz_reparam,
        material_name=args.material_name,
        measured_material_path=args.measured_material_path,
        ml_theta_res=args.ml_theta_res,
        ml_phi_res=args.ml_phi_res,
        dhr_theta_res=args.dhr_theta_res,
        dhr_phi_res=args.dhr_phi_res,
        use_nd_metalogs=args.use_nd_metalogs,
        additional_theta_i=args.additional_theta_i,
        anisotropic=args.anisotropic,
    )

    # get the output directory for future use and initialize logger
    output_dir = Path(meta_material.get_output_dir(resources_dir))
    logger = setup_logging(output_dir / "logs")

    try:
        metalog_parameters = build_metalog_parameters(args)

        # --- Log Configuration ---
        config_log_path = output_dir / "logs" / "configuration.log"
        with open(config_log_path, "w") as f:
            args.measured_material_path = (
                args.measured_material_path.name
                if args.measured_material_path is not None
                else ""
            )
            full_config = {**vars(args), **metalog_parameters}
            full_config["output_dir"] = str(full_config["output_dir"])
            json.dump(full_config, f, indent=4)
        logger.info(f"Configuration saved to: {config_log_path}")

        # --- Create Metalog Material ---
        logger.info("Starting Metalog BSDF creation...")
        logger.info(f"Marginal mode: {args.marginal_mode}")
        sample_metalogs_dict, brdf_dict = meta_material.make_metalog_material(
            metalog_parameters, make_probs_method=args.make_probs_method
        )

        # --- Save Distributions ---
        meta_material.save_distributions(resources_dir, sample_metalogs_dict)
        meta_material.save_directional_hemispherical_reflectance(
            brdf_dict, resources_dir
        )

        # --- Plotting (optional) ---
        plots_dir = output_dir / "plots"
        if not args.no_plots:
            generate_plots(
                args,
                meta_material,
                plots_dir,
                logger,
                sample_metalogs_dict,
            )

        logger.info("Script finished successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during script execution: {e}")
        raise


if __name__ == "__main__":
    main()
