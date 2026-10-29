# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

from pathlib import Path
import logging
import numpy as np
from datetime import datetime, timezone
import sys
import os

from tqdm import tqdm


def init_worker_process(gpu_queue):
    """
    Called exactly once per worker process when the pool starts.
    """

    # get a unique GPU ID from the queue
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def recursive_to_cpu(obj):
    """Recursively move all tensors in a data structure to the CPU."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cpu(v) for v in obj]
    return obj


def fit_mixture_metalog_material_from_mitsuba_bsdf_slice(args):
    """
    Worker entry point for parallel metalog fitting.

    Parameters (packed in args):
        conditional_mode: "metalog" | "mixture"
        make_probs_method: str
        metalog_marginal: "phi" | "cos_theta"
        metalog_parameters: dict
        marginal_bounds: ...
        conditional_bounds: ...
        plots_dir: str or Path
        measurement_dir_label: str
        sample2_np: np.ndarray of shape (N, 2)
        log_info: str
    """
    (
        conditional_mode,
        make_probs_method,
        metalog_marginal,
        metalog_parameters,
        marginal_bounds,
        conditional_bounds,
        plots_dir,
        measurement_dir_label,
        sample2_np,
        log_info,
        torch_seed,
    ) = args

    # local imports
    import torch
    from lib.utils.io_utils import log_conditional_metalog_infeasibility
    from lib.fit.mixture_models.marginal_conditional_mixture_metalog_maker import (
        make_marginal_mixture_conditional_metalogs_samples,
        make_marginal_bivariate_metalogs_samples,
    )

    is_cuda_available = torch.cuda.is_available()
    if torch_seed is not None:
        np.random.seed(torch_seed)

        if is_cuda_available:
            torch.cuda.manual_seed(torch_seed)

    if is_cuda_available:
        # with masking the assigned GPU is always index 0
        torch.cuda.set_device(0)
        device = "cuda"
    else:
        device = "cpu"

    # move data to this worker's device
    sample2 = torch.from_numpy(sample2_np).to(
        device=device, dtype=torch.float32
    )

    if conditional_mode == "metalog":
        sample_metalogs_dict = (
            make_marginal_mixture_conditional_metalogs_samples(
                sample2,
                device,
                metalog_parameters["min_regularization_term_marginal"],
                metalog_parameters["max_regularization_term_marginal"],
                metalog_parameters["min_regularization_term_conditional"],
                metalog_parameters["min_metalog_res_marg"],
                metalog_parameters["max_metalog_res_marg"],
                metalog_parameters["min_metalog_res_cond"],
                metalog_parameters["max_metalog_res_cond"],
                metalog_parameters["max_num_terms"],
                metalog_marginal,
                plots_dir=plots_dir,
                marginal_bounds=marginal_bounds,
                conditional_bounds=conditional_bounds,
                make_probs_method=make_probs_method,
                measurement_dir_label=measurement_dir_label,
                mix_K=metalog_parameters["mix_k_marg"],
                mix_alpha=metalog_parameters["mix_alpha"],
                mix_max_iters=metalog_parameters["mix_max_iters"],
                mixture_model=metalog_parameters["mix_model"],
                mix_metalog_term_marg=metalog_parameters[
                    "mix_metalog_term_marg"
                ],
                n_conditionals=metalog_parameters["n_conditionals"],
            )
        )

        log_conditional_metalog_infeasibility(
            metalog_parameters["n_conditionals"],
            metalog_parameters["max_num_terms"],
            log_info,
            sample_metalogs_dict["conditional"],
        )
    else:
        sample_metalogs_dict = make_marginal_bivariate_metalogs_samples(
            sample2,
            device,
            metalog_parameters["min_regularization_term_marginal"],
            metalog_parameters["max_regularization_term_marginal"],
            metalog_parameters["min_regularization_term_conditional"],
            metalog_parameters["max_regularization_term_conditional"],
            metalog_parameters["min_metalog_res_marg"],
            metalog_parameters["max_metalog_res_marg"],
            metalog_parameters["min_metalog_res_cond"],
            metalog_parameters["max_metalog_res_cond"],
            metalog_marginal,
            plots_dir=plots_dir,
            marginal_bounds=marginal_bounds,
            conditional_bounds=conditional_bounds,
            make_probs_method=make_probs_method,
            measurement_dir_label=measurement_dir_label,
            mix_K_marg=metalog_parameters["mix_k_marg"],
            mix_K_cond=metalog_parameters["mix_k_cond"],
            mix_metalog_term_marg=metalog_parameters["mix_metalog_term_marg"],
            mix_metalog_term_cond=metalog_parameters["mix_metalog_term_cond"],
            mix_alpha=metalog_parameters["mix_alpha"],
            mix_max_iters=metalog_parameters["mix_max_iters"],
            mixture_model=metalog_parameters["mix_model"],
            n_conditionals=metalog_parameters["n_conditionals"],
        )

    sample_metalogs_dict = recursive_to_cpu(sample_metalogs_dict)

    return measurement_dir_label, sample_metalogs_dict


def evaluate_dhr_slice(args):
    """
    Worker entry point for parallel DHR evaluation.
    """
    mi_bsdf_dict, theta_i, phi_i, theta_idx, phi_idx, n_samples = args

    # local imports
    import mitsuba as mi
    import drjit as dr
    import torch
    import numpy as np

    if torch.cuda.is_available():
        # with masking the assigned GPU is always index 0
        torch.cuda.set_device(0)
        mi.set_variant("cuda_ad_rgb")
        device = "cuda"
    else:
        device = "cpu"
        mi.set_variant("scalar_rgb")

    """
    Evaluate the directional hemispherical reflectance (DHR).
    DHR is denoted as:

        rho(w_i) = \\Int f_r(w_i, w_o) cos(theta_o) dw_o

    For an arbitrary function g(x), the general Monte Carlo estimator is:

        \\Int g(x) dx ~= 1/N \\sum_{k=1}^{N} g(X_k)/p(X_k),
    
    where we draw N random samples X_k from a PDF p(x). Appying this to DHR, we
    define g(w_o) as our target function:

        g(w_o) = f_r(w_i, w_o) cos(theta_o).
    
    Mitsuba's sample function returns the Monte Carlo importance weight:

        w = f_r(w_i, w_o) cos(theta_o) / p(w_o).

    Hence to precalculate the true DHR, we can take the expected value of w:

        E[w] = \\Int w p(w_o) dw_o 
             = \\Int [f_r(w_i, w_o) cos(theta_o) / p(w_o)] p(w_o) dw_o
             = \\Int f_r(w_i, w_o) cos(theta_o) dw_o
             = rho(w_i).
    
    Note thatthe expected value of the quotient is exactly the original
    integral, hence unbiased.
    """

    # reload the BSDF in this process
    mi_bsdf = mi.load_dict(mi_bsdf_dict)

    def sph_to_dir_local(theta, phi):
        st, ct = dr.sincos(theta)
        sp, cp = dr.sincos(phi)
        return mi.Vector3f(cp * st, sp * st, ct)

    # generate uniform random numbers for importance sampling
    unif = mi.Float(torch.rand(n_samples, device=device))
    unif2 = torch.rand((n_samples, 2), device=device)
    unif2 = mi.Point2f(unif2.T)

    # create dummy surface interaction
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = sph_to_dir_local(mi.Float(theta_i), mi.Float(phi_i))

    # bs_weight contains the albedo estimate: (f_r * cos_theta_o) / pdf
    bs_rec, bs_weight = mi_bsdf.sample(mi.BSDFContext(), si, unif, unif2)

    # filter out invalid samples
    valid_mask = bs_rec.wo.z > 0
    is_finite = (
        dr.isfinite(bs_weight.x)
        & dr.isfinite(bs_weight.y)
        & dr.isfinite(bs_weight.z)
    )
    bs_weight = dr.select(valid_mask & is_finite, bs_weight, 0.0)

    # the DHR is simply the average of the sample weights
    albedo_r = dr.mean(bs_weight.x)[0]
    albedo_g = dr.mean(bs_weight.y)[0]
    albedo_b = dr.mean(bs_weight.z)[0]

    # clamp the final integrated DHR to valid physical bounds [0.0, 1.0]
    dhr_result = np.clip([albedo_r, albedo_g, albedo_b], 0.0, 1.0).tolist()

    return theta_idx, phi_idx, dhr_result


class MetaBSDFMaker:

    COS_THETA_BOUNDS = (0.0, 1.0)

    # for multivariate metalogs
    ml_phi_i_default = [
        -3.1415927,
        -2.91717,
        -2.6606486,
        -2.2855954,
        -1.5707964,
        -0.9063681,
        -0.47562298,
        -0.20649175,
        0.0,
        0.26754296,
        0.5205299,
        0.864391,
        1.5707964,
        2.3013651,
        2.7094657,
        2.950614,
        3.1415927,
    ]
    ml_theta_i_default = [
        0.0,
        0.5220543,
        0.7482307,
        0.9298108,
        1.091006,
        1.2424256,
        1.3924501,
        1.5707963,
    ]

    def __init__(
        self,
        mi_bsdf_dict: dict,
        metalog_marginal: str,
        is_bounded: bool,
        resources_dir: str,
        marginal_mode: str = "metalog",
        conditional_mode: str = "metalog",
        reparam: bool = False,
        material_name: str | None = None,
        measured_material_path: Path | None = None,
        ml_theta_res: int = -1,
        ml_phi_res: int = -1,
        dhr_theta_res: int = 32,
        dhr_phi_res: int = 16,
        use_nd_metalogs: bool = True,
        ml_theta_i: list | None = None,
        additional_theta_i: bool = False,
        anisotropic: bool = False,
    ):
        # local imports
        import mitsuba as mi
        import torch

        """
        Not great but
        That's life (that's life)
        That's what all the people say
        You're riding high in April, shot down in May
        ...
        """
        sys.path.append("extern/brdf-loader/python")
        import visualize as mvis

        self.use_nd_metalogs = use_nd_metalogs

        if marginal_mode not in ("metalog", "mixture"):
            raise ValueError("marginal_mode must be 'metalog' or 'mixture'")
        self.marginal_mode = marginal_mode

        if conditional_mode not in ("metalog", "mixture"):
            raise ValueError("conditional_mode must be 'metalog' or 'mixture'")
        self.conditional_mode = conditional_mode

        # initialize utc date and time (relevant for `get_output_dir()`)
        self.utc_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self.utc_time = (
            datetime.now(tz=timezone.utc).time().strftime("%H-%M-%S")
        )

        # initialize torch device and mitsuba variant
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        mi_variant = "cuda_ad_rgb" if self.device == "cuda" else "scalar_rgb"
        mi.set_variant(mi_variant)

        self.mi_bsdf_dict = mi_bsdf_dict
        self.material_name = material_name
        self.measured_material_name = (
            measured_material_path.name
            if measured_material_path is not None
            else None
        )
        self.rusinkiewicz_reparam = reparam
        self.metalog_marginal = metalog_marginal
        self.is_bounded = is_bounded
        self.is_isotropic = not anisotropic

        # initialize incident azimuth angles
        if self.rusinkiewicz_reparam and self.is_isotropic:
            # use Rusinkiewicz98 parameterization
            self.ml_phi_i = [0.0]
        else:
            if ml_phi_res < 0:
                self.ml_phi_i = self.ml_phi_i_default
            else:
                self.ml_phi_i = np.linspace(
                    self.ml_phi_i_default[0],
                    self.ml_phi_i_default[-1],
                    ml_phi_res,
                ).tolist()

        # initialize incident elevation angles
        if measured_material_path is not None:
            measured_bsdf_tensor = mvis.read_tensor(str(measured_material_path))
            self.ml_theta_i = measured_bsdf_tensor["theta_i"]

            """
            Add an additional theta_i when the delta between the first and
            second theta_i exceeds some fix threshold. We observed
            better quality in our renders.
            """
            delta_theta_01 = 0.3
            if (
                additional_theta_i
                and self.ml_theta_i[1] - self.ml_theta_i[0] > delta_theta_01
            ):
                self.ml_theta_i = self.ml_theta_i.tolist()
                self.ml_theta_i.append(0.16)
                self.ml_theta_i.append(0.32)
                self.ml_theta_i = np.array(sorted(self.ml_theta_i))

            """
            For some materials (e.g., vch-golden-yellow) we observed no valid
            samples for ml_theta_i[-1] (pi/2, grazing angle), hence we subtract 
            an epsilon for valid samples.
            """
            self.ml_theta_i = np.clip(self.ml_theta_i, 0, np.pi / 2 - 1e-4)

            # override azimuth angles if material anisotropic
            if len(measured_bsdf_tensor["phi_i"]) > 1:
                self.ml_phi_i_default = self.ml_phi_i = measured_bsdf_tensor[
                    "phi_i"
                ]

            # override azimuth angles if resolution is not negative
            if ml_phi_res > 0:
                self.ml_phi_i = np.linspace(
                    self.ml_phi_i_default[0],
                    self.ml_phi_i_default[-1],
                    ml_phi_res,
                ).tolist()
        else:
            self.ml_theta_i = self.ml_theta_i_default.copy()

            """
            Add an additional theta_i when the delta between the first and
            second theta_i exceeds some fix threshold. We observed
            better quality in our renders.
            """
            delta_theta_01 = 0.3
            if (
                additional_theta_i
                and self.ml_theta_i[1] - self.ml_theta_i[0] > delta_theta_01
            ):
                self.ml_theta_i.append(0.16)
                self.ml_theta_i.append(0.32)
                self.ml_theta_i = np.array(sorted(self.ml_theta_i))

        # override elevation angles if resolution is not negative
        if ml_theta_res > 0:
            if ml_theta_i is not None and len(ml_theta_i) == ml_theta_res:
                self.ml_theta_i = ml_theta_i
            else:
                cos_theta_steps = np.linspace(1.0, 1e-4, ml_theta_res)
                np.arccos(cos_theta_steps).tolist()
                self.ml_theta_i = np.arccos(cos_theta_steps).tolist()

        if len(self.ml_theta_i) < 1:
            raise ValueError("Incident elevation angle resolution is 0.")
        if len(self.ml_phi_i) < 1:
            raise ValueError("Incident azimuth angle resolution is 0.")

        # for storing directional-hemispherical reflectance
        self.dhr_phi_i = np.linspace(
            -3.1415927, 3.1415927, dhr_phi_res
        ).tolist()
        # linear steps of cos(theta) from 0.0 (equator) to 1.0 (pole)
        cos_theta_steps = np.linspace(1.0, 1e-4, dhr_theta_res)
        self.dhr_theta_i_cosine_spaced = np.arccos(cos_theta_steps).tolist()

        # load Mitsuba BSDF
        self.mi_bsdf = mi.load_dict(self.mi_bsdf_dict)

        self.plots_dir = None
        if resources_dir != None:
            output_dir = self.get_output_dir(resources_dir)
            self.plots_dir = Path(f"{output_dir}/plots")
            self.plots_dir.mkdir(parents=True, exist_ok=True)

    def make_metalog_material(
        self,
        metalog_parameters: dict,
        phi_incident_angles: list | None = None,
        make_probs_method: str = "classic",
        torch_seed: int | None = 42,
    ) -> dict:
        if phi_incident_angles is None:
            phi_incident_angles = self.ml_phi_i

        logging.info(f"Evaluating directional-hemispherical reflectance.")
        result_dhr = self.make_directional_hemispherical_reflectance(
            metalog_parameters["n_samples"]
        )

        logging.info(f"Fitting metalogs for sampling.")
        theta_incident_angles = self.ml_theta_i
        if not self.use_nd_metalogs:
            phi_incident_angles = theta_incident_angles = [0.0]

        if self.marginal_mode == "metalog":
            result_sample = self.make_metalog_material_from_mistuba_bsdf(
                metalog_parameters,
                phi_incident_angles,
                theta_incident_angles,
                make_probs_method,
                torch_seed,
            )
        else:
            result_sample = (
                self.make_mixture_metalog_material_from_mistuba_bsdf(
                    metalog_parameters,
                    phi_incident_angles,
                    theta_incident_angles,
                    make_probs_method,
                    torch_seed,
                )
            )

        return (result_sample, result_dhr)

    def make_metalog_material_from_mistuba_bsdf(
        self,
        metalog_parameters,
        phi_incident_angles,
        theta_incident_angles,
        make_probs_method,
        torch_seed: int | None = 42,
    ):
        # local imports
        import torch
        import mitsuba as mi
        import drjit as dr
        from lib.utils.sample_record import SampleRecord
        from lib.metalogs import (
            unitvec_to_cos_theta_phi,
            make_marginal_conditional_metalogs,
        )
        from lib.utils.io_utils import (
            log_conditional_metalog_infeasibility,
            log_marginal_metalog_infeasability,
        )

        if torch_seed is not None:
            if torch.cuda.is_available():
                torch.cuda.manual_seed(torch_seed)

        result_sample = {}

        marginal_bounds, conditional_bounds = self.get_bounds()

        # create uniform samples
        n_samples = metalog_parameters["n_samples"]
        unif = mi.Float(torch.rand(n_samples, device=self.device))
        unif2 = torch.rand((n_samples, 2), device=self.device)
        unif2_dr = mi.Point2f(unif2.T)

        # create a (dummy) surface interaction to use for sampling
        si = dr.zeros(mi.SurfaceInteraction3f)

        """
        For each measured incident direction sample the Mitsuba measured
        material and then create a bivariate metalog.
        """
        for phi_i in phi_incident_angles:
            for theta_i in theta_incident_angles:
                measurement_dir_label = self.measurement_dir_label(
                    float(phi_i), float(theta_i)
                )
                log_info = f"{measurement_dir_label}"

                # get the incident direction
                si.wi = self.sph_to_dir(theta_i, phi_i)
                logging.info(
                    f"2D: ({dr.rad2deg(theta_i)}, {dr.rad2deg(phi_i)}), 3D:{si.wi}"
                )

                # sample the measured material for the incident direction
                bs_rec, bs_value = self.mi_bsdf.sample(
                    mi.BSDFContext(),
                    si,
                    unif,
                    unif2_dr,
                )
                p_wo = bs_rec.pdf.numpy().reshape(-1)

                # compute valid wo
                valid_mask = bs_rec.wo.z > 0
                idx_valid = dr.compress(valid_mask)
                wi_valid = dr.gather(mi.Vector3f, si.wi, idx_valid)
                wo_valid = dr.gather(mi.Vector3f, bs_rec.wo, idx_valid)

                if self.rusinkiewicz_reparam:
                    # calculate the halfway vector
                    sample3 = dr.normalize(wo_valid + wi_valid).torch().T
                else:
                    sample3 = dr.normalize(wo_valid).torch().T

                sample2 = unitvec_to_cos_theta_phi(sample3)
                if self.metalog_marginal == "phi":
                    sample2 = sample2.flip(1)

                # compute bivariate metalog for sampling wo
                sRec = SampleRecord(
                    unif2=unif2,
                    pdf=torch.from_numpy(p_wo),
                    samples=sample3,
                )
                sample_metalogs_dict = make_marginal_conditional_metalogs(
                    sample2,
                    self.device,
                    metalog_parameters["min_regularization_term_marginal"],
                    metalog_parameters["min_regularization_term_conditional"],
                    metalog_parameters["min_metalog_res_marg"],
                    metalog_parameters["min_metalog_res_cond"],
                    metalog_parameters["max_num_terms"],
                    marginal_bounds=marginal_bounds,
                    conditional_bounds=conditional_bounds,
                    make_probs_method=make_probs_method,
                    n_conditionals=metalog_parameters["n_conditionals"],
                )
                sample_metalogs_dict["data_pdf"] = sRec.get("pdf")
                sample_metalogs_dict["metalog_marginal"] = self.metalog_marginal

                # log conditional metalog infeasability
                log_conditional_metalog_infeasibility(
                    metalog_parameters["n_conditionals"],
                    metalog_parameters["max_num_terms"],
                    log_info,
                    sample_metalogs_dict["conditional"],
                )

                # save metalogs for sampling wo/h
                result_sample[measurement_dir_label] = sample_metalogs_dict

                # log marginal metalog infeasability and error
                log_marginal_metalog_infeasability(
                    metalog_parameters["max_num_terms"],
                    log_info,
                    sample_metalogs_dict["marginal"],
                )
                logging.info(
                    f"({log_info}/wo) Marginal metalog errors: \n{sample_metalogs_dict["marginal_score"]}"
                )

        return result_sample

    def make_mixture_metalog_material_from_mistuba_bsdf(
        self,
        metalog_parameters,
        phi_incident_angles,
        theta_incident_angles,
        make_probs_method,
        torch_seed: int | None = 42,
    ):
        # local imports
        import torch
        import torch.multiprocessing as mp
        import mitsuba as mi
        import drjit as dr
        from lib.utils.sample_record import SampleRecord
        from lib.metalogs import unitvec_to_cos_theta_phi

        if torch_seed is not None:
            torch.manual_seed(torch_seed)
            torch.cuda.manual_seed_all(torch_seed)

        result_sample = {}
        marginal_bounds, conditional_bounds = self.get_bounds()

        # create uniform samples
        n_samples = metalog_parameters["n_samples"]
        unif = mi.Float(torch.rand(n_samples, device=self.device))
        unif2 = torch.rand((n_samples, 2), device=self.device)
        unif2_dr = mi.Point2f(unif2.T)

        # create a (dummy) surface interaction to use for sampling
        si = dr.zeros(mi.SurfaceInteraction3f)

        n_gpu = torch.cuda.device_count()
        use_pool = self.device == "cuda" and n_gpu > 1
        tasks = []

        """
        For each measured incident direction sample the Mitsuba measured
        material and then create a bivariate metalog.
        """
        for phi_i in phi_incident_angles:
            for theta_i in theta_incident_angles:
                measurement_dir_label = self.measurement_dir_label(
                    float(phi_i), float(theta_i)
                )
                log_info = f"{measurement_dir_label}"

                # get the incident direction
                si.wi = self.sph_to_dir(theta_i, phi_i)

                # sample the measured material for the incident direction
                bs_rec, bs_value = self.mi_bsdf.sample(
                    mi.BSDFContext(),
                    si,
                    unif,
                    unif2_dr,
                )

                # compute valid wo
                valid_mask = bs_rec.wo.z > 0
                idx_valid = dr.compress(valid_mask)
                wi_valid = dr.gather(mi.Vector3f, si.wi, idx_valid)
                wo_valid = dr.gather(mi.Vector3f, bs_rec.wo, idx_valid)

                if self.rusinkiewicz_reparam:
                    # calculate the halfway vector
                    sample3 = dr.normalize(wo_valid + wi_valid).torch().T
                else:
                    sample3 = dr.normalize(wo_valid).torch().T

                sample2 = unitvec_to_cos_theta_phi(sample3)
                if self.metalog_marginal == "phi":
                    sample2 = sample2.flip(1)

                # compute bivariate metalog for sampling wo
                sRec = SampleRecord(
                    unif2=unif2,
                    pdf=bs_rec.pdf.torch(),
                    samples=sample3,
                )

                # move input data to numpy for safe pickling into subprocesses
                sample2_np = (
                    sample2.detach().cpu().numpy()
                    if isinstance(sample2, torch.Tensor)
                    else np.array(sample2)
                )

                # enqueue this metalog-fitting job
                tasks.append(
                    (
                        self.conditional_mode,
                        make_probs_method,
                        self.metalog_marginal,
                        metalog_parameters,
                        marginal_bounds,
                        conditional_bounds,
                        self.plots_dir,
                        measurement_dir_label,
                        sample2_np,
                        log_info,
                        torch_seed,
                    )
                )

        # dispatch tasks
        if use_pool and len(tasks) > 0:
            logging.info(
                f"Distributing mixture metalog fitting across {n_gpu} GPUs."
            )
            ctx = mp.get_context("spawn")

            # create a queue and fill it with GPU IDs
            gpu_queue = ctx.Queue()
            for i in range(n_gpu):
                gpu_queue.put(i)

            # pass the queue to the initializer
            with ctx.Pool(
                processes=n_gpu,
                initializer=init_worker_process,
                initargs=(gpu_queue,),
            ) as pool:
                iterator = pool.imap_unordered(
                    fit_mixture_metalog_material_from_mitsuba_bsdf_slice, tasks
                )

                for (
                    measurement_dir_label,
                    sample_metalogs_dict,
                ) in tqdm(
                    iterator, total=len(tasks), desc="Fitting (Multi-GPU)"
                ):
                    result_sample[measurement_dir_label] = sample_metalogs_dict
        else:
            # single-GPU / CPU path
            for args in tqdm(tasks, desc="Fitting (Single Process)"):
                measurement_dir_label, sample_metalogs_dict = (
                    fit_mixture_metalog_material_from_mitsuba_bsdf_slice(args)
                )
                result_sample[measurement_dir_label] = sample_metalogs_dict

        logging.info(f"Completed mixture metalog fitting.")

        return result_sample

    def make_directional_hemispherical_reflectance(
        self, n_samples: int = 512000
    ) -> np.ndarray:
        """Evaluate and store directional-hemispherical reflectance into numpy tensor."""
        import torch
        import torch.multiprocessing as mp

        """
        Prepare a tensor of [height, width, channels], where height is the theta-axis (rows), width is the phi-axis
        (height) and channels are RGB.
        """
        height = len(self.dhr_theta_i_cosine_spaced)
        width = len(self.dhr_phi_i)
        result = np.zeros((height, width, 3), dtype=np.float32)

        logging.info(f"Evaluating DHR grid: {width}x{height}")

        n_gpu = torch.cuda.device_count()
        use_pool = self.device == "cuda" and n_gpu > 1
        tasks = []

        """
        For each incident directions evaluate and store the
        directional-hemispherical reflectance.
        """
        for phi_idx, phi_i in enumerate(self.dhr_phi_i):
            for theta_idx, theta_i in enumerate(self.dhr_theta_i_cosine_spaced):
                tasks.append(
                    (
                        self.mi_bsdf_dict,
                        float(theta_i),
                        float(phi_i),
                        theta_idx,
                        phi_idx,
                        n_samples,
                    )
                )

        if use_pool and len(tasks) > 0:
            logging.info(f"Distributing DHR evaluation across {n_gpu} GPUs.")
            ctx = mp.get_context("spawn")

            # create a queue and fill it with GPU IDs
            gpu_queue = ctx.Queue()
            for i in range(n_gpu):
                gpu_queue.put(i)

            # pass the queue to the initializer
            with ctx.Pool(
                processes=n_gpu,
                initializer=init_worker_process,
                initargs=(gpu_queue,),
            ) as pool:
                iterator = pool.imap_unordered(evaluate_dhr_slice, tasks)

                for theta_idx, phi_idx, dhr_val in tqdm(
                    iterator, total=len(tasks), desc="Eval DHR (Multi-GPU)"
                ):
                    result[theta_idx, phi_idx, :] = dhr_val
        else:
            # single-GPU / CPU path
            for args in tqdm(tasks, desc="Eval DHR (Single Process)"):
                theta_idx, phi_idx, dhr_val = evaluate_dhr_slice(args)
                result[theta_idx, phi_idx, :] = dhr_val

        logging.info(f"Completed DHR evaluation.")

        return result

    def save_distributions(
        self,
        resources_dir: str,
        metalogs_dict: dict,
        phi_incident_angles: list | None = None,
    ):
        # local imports
        import torch
        from lib.metalogs import save_marginal_conditional_metalogs

        if self.metalog_marginal is None:
            raise ValueError("Metalog marginal is not initialized.")
        if self.is_bounded is None:
            raise ValueError("Boundedness is not initialized.")

        if not self.use_nd_metalogs:
            assert (
                len(metalogs_dict) == 1
            ), "`use_nd_metalogs=True`: `len(metalogs_dict)` must be 1."
            phi_incident_angles = theta_incident_angles = [0.0]
        else:
            theta_incident_angles = self.ml_theta_i

        def save(output_dir, metalogs_dict):
            if self.rusinkiewicz_reparam:
                # not necessary
                return

            for measurement_dir_label, metalogs in metalogs_dict.items():
                final_output_dir = (
                    f"{output_dir}/{measurement_dir_label}/sample_metalogs"
                )
                final_output_dir = os.path.abspath(final_output_dir)
                save_marginal_conditional_metalogs(metalogs, final_output_dir)

        def save_compact(
            output_dir,
            metalogs_dict,
            phi_incident_angles,
            theta_incident_angles,
        ):
            compact_result = {}

            if phi_incident_angles is None:
                phi_incident_angles = self.ml_phi_i

            for phi_i in phi_incident_angles:
                for theta_i in theta_incident_angles:
                    measurement_dir_label = self.measurement_dir_label(
                        float(phi_i), float(theta_i)
                    )
                    metalogs = metalogs_dict[measurement_dir_label]

                    entry = {}
                    if "marginal" in metalogs:
                        entry["marginal"] = metalogs["marginal"]
                    if "marginal_mixture" in metalogs:
                        entry["marginal_mixture"] = metalogs["marginal_mixture"]
                        entry["marginal_inverse_cdf"] = metalogs[
                            "marginal_inverse_cdf"
                        ]
                        entry["marginal_quantile_cdf_range"] = metalogs[
                            "marginal_quantile_cdf_range"
                        ]
                    if "conditional_mixture" in metalogs:
                        entry["conditional_mixture"] = metalogs[
                            "conditional_mixture"
                        ]
                    if "conditional" in metalogs:
                        # always present except rusinkiewicz_reparam=true
                        entry["conditional"] = metalogs["conditional"]

                    # early exit if we are using a single slice
                    if not self.use_nd_metalogs:
                        assert len(metalogs_dict) == 1
                        torch.save(
                            entry,
                            os.path.abspath(f"{output_dir}/sample_metalogs.pt"),
                        )
                        return

                    compact_result[(float(phi_i), float(theta_i))] = entry

                torch.save(
                    compact_result,
                    os.path.abspath(f"{output_dir}/sample_metalogs.pt"),
                )

        output_dir = self.get_output_dir(resources_dir)

        if self.marginal_mode == "metalog":
            save(output_dir, metalogs_dict)
        save_compact(
            output_dir,
            metalogs_dict,
            phi_incident_angles,
            theta_incident_angles,
        )

        logging.log(logging.INFO, f"Saved metalogs in {output_dir}.")

    def save_directional_hemispherical_reflectance(
        self,
        data: dict,
        resources_dir: str,
    ):
        import mitsuba as mi

        output_dir = self.get_output_dir(resources_dir)
        output_path = f"{output_dir}/brdf.exr"

        # create a mitsuba bitmap from the numpy array and save it as EXR
        mi.Bitmap(data).write(output_path)
        logging.info(f"Saved directional albedo to {output_path}")

    def measurement_dir_label(self, phi_i, theta_i, round_decimals=2):
        phi_i_rounded = np.round(phi_i, decimals=round_decimals)
        theta_i_rounded = np.round(theta_i, decimals=round_decimals)
        measurement_dir_label = (
            f"phi_i_{phi_i_rounded}_theta_i_{theta_i_rounded}"
        )

        return measurement_dir_label

    def get_bounds(self):
        import torch

        if not self.is_bounded:
            return None, None

        phi_bounds = (-torch.pi, torch.pi)

        if self.metalog_marginal == "cos_theta":
            marginal_bounds = self.COS_THETA_BOUNDS
            conditional_bounds = phi_bounds
        else:
            marginal_bounds = phi_bounds
            conditional_bounds = self.COS_THETA_BOUNDS

        return marginal_bounds, conditional_bounds

    def get_output_dir(self, resources_dir):
        is_bounded_str = "bounded" if self.is_bounded else "unbounded"
        bounded_dir = f"{resources_dir}/{is_bounded_str}"
        reparam_label = "_reparam" if self.rusinkiewicz_reparam else ""
        measured_material_label = (
            f"_{self.measured_material_name.split(".")[0]}"
            if self.measured_material_name != None
            and self.measured_material_name != "None"
            else ""
        )
        sample_type_dir = f"{bounded_dir}/{self.material_name}{reparam_label}{measured_material_label}"
        metalog_marginal_dir = f"{sample_type_dir}/{self.metalog_marginal}"
        utc_date_dir = f"{metalog_marginal_dir}/{self.utc_date}"
        utc_time_dir = f"{utc_date_dir}/{self.utc_time}"
        output_dir = f"{utc_time_dir}"
        return output_dir

    def sph_to_dir(self, theta, phi):
        """Map spherical to Euclidean coordinates"""
        import mitsuba as mi
        import drjit as dr

        st, ct = dr.sincos(theta)
        sp, cp = dr.sincos(phi)
        return mi.Vector3f(cp * st, sp * st, ct)
