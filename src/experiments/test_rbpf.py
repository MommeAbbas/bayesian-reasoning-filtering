import numpy as np

from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator, Mode
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.rbpf_slds import RBPFConfig, RBPF_SLDS


def main():
    np.random.seed(0)

    # Dynamics: keep noise small for a clear test
    dcfg = SLDSConfig(noise_std=(0.01, 0.01, 0.01))
    dyn = SLDSDynamics(dcfg)
    sim = SLDSSimulator(dyn)

    # Sensors: moderate noise, some outliers
    scfg = SensorConfig(noise_std=(0.05, 0.05, 0.05), outlier_prob=0.05, outlier_scale=25.0)
    sensors = ReasoningSensors(scfg)

    # Simulate ground truth
    T = 40
    xs, zs = sim.run(T=T, x0=np.array([0.2, 0.5, 0.8]), z0=Mode.NORMAL)

    # RBPF
    fcfg = RBPFConfig(num_particles=300, resample_threshold=0.5)
    rbpf = RBPF_SLDS(dyn=dyn, sensors=sensors, cfg=fcfg)

    print("k  true_mode  true_p  est_p  mode_probs[N,I,B]")
    for k in range(1, T + 1):
        y = sensors.observe(xs[k])
        x_hat, mode_probs = rbpf.step(y)

        true_mode = Mode(int(zs[k])).name
        print(f"{k:02d} {true_mode:9s} {xs[k,0]:.3f}  {x_hat[0]:.3f}  "
              f"[{mode_probs[0]:.2f},{mode_probs[1]:.2f},{mode_probs[2]:.2f}]")


if __name__ == "__main__":
    main()
