import time
import numpy as np

from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator
from src.simulation.sensors import SensorConfig, ReasoningSensors

from src.filters.rbpf_slds import RBPF_SLDS, RBPFConfig
from src.filters.pf_baseline import PF_SLDS, PFConfig

from src.evaluation.online_prediction import OnlineCorrectnessPredictor
from src.evaluation.metrics import (
    negative_log_likelihood,
    brier_score,
    expected_calibration_error,
    auc_by_prefix,
)


def run_single_trajectory_rbpf(T, dyn, sensors, rbpf, predictor):
    xs, _ = SLDSSimulator(dyn).run(T=T)
    c = int(xs[-1, 0] > 0.6)

    preds = []

    for k in range(1, T + 1):
        y = sensors.observe(xs[k])
        rbpf.step(y)
        p_hat = predictor.prob_correct_from_particles(rbpf.mus, rbpf.weights)
        preds.append(p_hat)

    return np.array(preds), c


def run_single_trajectory_pf(T, dyn, sensors, pf, predictor):
    xs, _ = SLDSSimulator(dyn).run(T=T)
    c = int(xs[-1, 0] > 0.6)

    preds = []

    for k in range(1, T + 1):
        y = sensors.observe(xs[k])
        x_hat, _ = pf.step(y)
        p_hat = predictor.prob_correct_from_state(x_hat)
        preds.append(p_hat)

    return np.array(preds), c


def compute_metrics(all_preds, all_labels, T):
    return {
        "nll": negative_log_likelihood(
            all_preds.flatten(),
            np.repeat(all_labels, T),
        ),
        "brier": brier_score(
            all_preds.flatten(),
            np.repeat(all_labels, T),
        ),
        "ece": expected_calibration_error(
            all_preds.flatten(),
            np.repeat(all_labels, T),
        ),
        "auc_early": auc_by_prefix(
            [all_preds[:, k] for k in range(T)],
            all_labels,
        )[4],  # k = 5
    }


def main():
    np.random.seed(0)

    # Fixed experiment parameters
    T = 30
    N_traj = 80
    N_trials = 2  # average over seeds

    # Particle grids
    RBPF_particles = [25, 50, 100, 200, 300]
    PF_particles = [100, 300, 600, 1000]

    # Dynamics + sensors (fixed)
    dyn = SLDSDynamics(SLDSConfig())
    sensors = ReasoningSensors(
        SensorConfig(
            noise_std=(0.05, 0.05, 0.05),
            outlier_prob=0.05,
            outlier_scale=25.0,
        )
    )

    predictor = OnlineCorrectnessPredictor()

    results = {
        "RBPF": {},
        "PF": {},
    }

    # RBPF sweep
    for Np in RBPF_particles:
        print(f"[RBPF] particles = {Np}")
        metrics_list = []
        runtimes = []

        for trial in range(N_trials):
            all_preds = []
            all_labels = []

            t0 = time.perf_counter()

            for _ in range(N_traj):
                rbpf = RBPF_SLDS(
                    dyn=dyn,
                    sensors=sensors,
                    cfg=RBPFConfig(num_particles=Np),
                )

                preds, label = run_single_trajectory_rbpf(
                    T, dyn, sensors, rbpf, predictor
                )
                all_preds.append(preds)
                all_labels.append(label)

            runtime = time.perf_counter() - t0
            runtimes.append(runtime / N_traj)

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            metrics_list.append(
                compute_metrics(all_preds, all_labels, T)
            )

        results["RBPF"][Np] = {
            "metrics": metrics_list,
            "runtime_per_traj": runtimes,
        }

    # PF sweep
    for Np in PF_particles:
        print(f"[PF] particles = {Np}")
        metrics_list = []
        runtimes = []

        for trial in range(N_trials):
            all_preds = []
            all_labels = []

            t0 = time.perf_counter()

            for _ in range(N_traj):
                pf = PF_SLDS(
                    dyn=dyn,
                    sensors=sensors,
                    cfg=PFConfig(num_particles=Np),
                )

                preds, label = run_single_trajectory_pf(
                    T, dyn, sensors, pf, predictor
                )
                all_preds.append(preds)
                all_labels.append(label)

            runtime = time.perf_counter() - t0
            runtimes.append(runtime / N_traj)

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            metrics_list.append(
                compute_metrics(all_preds, all_labels, T)
            )

        results["PF"][Np] = {
            "metrics": metrics_list,
            "runtime_per_traj": runtimes,
        }

    np.save("particle_sweep_results.npy", results)
    print("Saved particle_sweep_results.npy")


if __name__ == "__main__":
    main()
