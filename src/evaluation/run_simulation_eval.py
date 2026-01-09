import numpy as np

from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.rbpf_slds import RBPFConfig, RBPF_SLDS
from src.filters.ekf_baseline import EKFBaseline, EKFConfig
from src.filters.pf_baseline import PF_SLDS, PFConfig
from src.evaluation.online_prediction import (
    OnlineCorrectnessPredictor,
    OnlinePredictionRecorder,
)
from src.evaluation.metrics import (
    negative_log_likelihood,
    brier_score,
    expected_calibration_error,
    auc_by_prefix,
)


def run_single_trajectory(T, dyn, sensors, rbpf, ekf, pf, predictor):
    xs, zs = SLDSSimulator(dyn).run(T=T)
    c = int(xs[-1, 0] > 0.6)

    rec_rbpf = OnlinePredictionRecorder()
    rec_ekf = OnlinePredictionRecorder()
    rec_pf = OnlinePredictionRecorder()

    for k in range(1, T + 1):
        y = sensors.observe(xs[k])

        # RBPF
        rbpf.step(y)
        p_rbpf = predictor.prob_correct_from_particles(rbpf.mus, rbpf.weights)
        rec_rbpf.update(p_rbpf)

        # EKF
        mu, Sigma = ekf.step(y)
        p_ekf = predictor.prob_correct_from_state(mu)
        rec_ekf.update(p_ekf)

        # PF
        x_hat_pf, _ = pf.step(y)
        p_pf = predictor.prob_correct_from_state(x_hat_pf)
        rec_pf.update(p_pf)

    return (
        rec_rbpf.as_array(),
        rec_ekf.as_array(),
        rec_pf.as_array(),
        c,
    )


def main():
    np.random.seed(0)

    # configuration
    T = 30
    N_traj = 200

    # Dynamics + sensors
    dyn = SLDSDynamics(SLDSConfig())
    sensors = ReasoningSensors(
        SensorConfig(
            noise_std=(0.05, 0.05, 0.05),
            outlier_prob=0.05,
            outlier_scale=25.0,
        )
    )

    predictor = OnlineCorrectnessPredictor()

    all_preds_rbpf = []
    all_preds_ekf = []
    all_preds_pf = []
    all_labels = []

    for n in range(N_traj):
        rbpf = RBPF_SLDS(
            dyn=dyn,
            sensors=sensors,
            cfg=RBPFConfig(num_particles=300),
        )
        ekf = EKFBaseline(
            dyn=dyn,
            sensors=sensors,
            cfg=EKFConfig(),
        )
        pf = PF_SLDS(
            dyn=dyn,
            sensors=sensors,
            cfg=PFConfig(num_particles=1500),
        )

        preds_rbpf, preds_ekf, preds_pf, label = run_single_trajectory(
            T=T,
            dyn=dyn,
            sensors=sensors,
            rbpf=rbpf,
            ekf=ekf,
            pf=pf,
            predictor=predictor,
        )

        all_preds_rbpf.append(preds_rbpf)
        all_preds_ekf.append(preds_ekf)
        all_preds_pf.append(preds_pf)
        all_labels.append(label)

    all_preds_rbpf = np.array(all_preds_rbpf)
    all_preds_ekf = np.array(all_preds_ekf)
    all_preds_pf = np.array(all_preds_pf)
    all_labels = np.array(all_labels)

    def compute_all_metrics(name, all_preds, all_labels, T):
        nll = negative_log_likelihood(
            p_hat=all_preds.flatten(),
            y_true=np.repeat(all_labels, T),
        )

        brier = brier_score(
            p_hat=all_preds.flatten(),
            y_true=np.repeat(all_labels, T),
        )

        ece = expected_calibration_error(
            p_hat=all_preds.flatten(),
            y_true=np.repeat(all_labels, T),
        )

        aucs = auc_by_prefix(
            predictions=[all_preds[:, k] for k in range(T)],
            labels=all_labels,
        )

        print(f"=== Evaluation results ({name}) ===")
        print(f"NLL:    {nll:.4f}")
        print(f"Brier: {brier:.4f}")
        print(f"ECE:    {ece:.4f}")
        print(f"AUC@final: {aucs[-1]:.4f}")
        print(f"AUC@early (k=5): {aucs[4]:.4f}")
        print()

    compute_all_metrics("RBPF", all_preds_rbpf, all_labels, T)
    compute_all_metrics("EKF",  all_preds_ekf,  all_labels, T)
    compute_all_metrics("PF",   all_preds_pf,   all_labels, T)


if __name__ == "__main__":
    main()
