# © 2025-2026 Saip-Can Hasbay, Hiroyuki Sakai

import mitsuba as mi
import drjit as dr
import numpy as np


def benchmark_mitsuba_scene_render(scene, spp, n_runs: int = 30, seed: int = 0):
    """
    Benchmarks a given DrJit function and returns the execution time.
    """
    # 1. Warmup call to compile the kernel, etc.
    dr.eval(
        mi.render(
            scene,
            spp=spp,
        )
    )

    # Store execution time of each run
    timings_ms = []

    # Run the benchmark n times
    for _ in range(n_runs):
        # Clear malloc cache and kernel history before each run
        dr.flush_malloc_cache()
        dr.kernel_history_clear()

        # Run the function with history recording enabled
        for _ in range(n_runs):
            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                dr.eval(
                    mi.render(
                        scene,
                        spp=spp,
                        seed=seed,
                    )
                )

        # Get execution time from kernel history for this run
        history = dr.kernel_history()
        if history:
            exec_time = sum(item["execution_time"] for item in history)
            timings_ms.append(exec_time)

    if not timings_ms:
        print("Warning: Kernel history was empty for all runs.")
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}

    # 3. Calculate statistics using NumPy
    # Convert list of timings from milliseconds to a NumPy array of seconds
    timings_s = np.array(timings_ms) * 1e-3

    stats = {
        "mean": np.mean(timings_s) / n_runs,
        "median": np.median(timings_s) / n_runs,
        "std": np.std(timings_s) / n_runs,
        "count": len(timings_s),
    }

    return stats
