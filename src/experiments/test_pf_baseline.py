import numpy as np

from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator, Mode
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.pf_baseline import PF_SLDS, PFConfig


def main():
    np.random.seed(0)

    dyn = SLDSDynamics(SLDSConfig())
    sensors = ReasoningSensors(SensorConfig(noise_std=(0.05, 0.05, 0.05), outlier_prob=0.05, outlier_scale=25.0))

    xs, zs = SLDSSimulator(dyn).run(T=25, x0=np.array([0.2, 0.5, 0.8]), z0=Mode.NORMAL)

    pf = PF_SLDS(dyn=dyn, sensors=sensors, cfg=PFConfig(num_particles=1000))

    print("k  true_mode  true_p  pf_p  mode_probs[N,I,B]")
    for k in range(1, 26):
        y = sensors.observe(xs[k])
        x_hat, mode_probs = pf.step(y)
        print(f"{k:02d} {Mode(int(zs[k])).name:9s} {xs[k,0]:.3f}  {x_hat[0]:.3f}  "
              f"[{mode_probs[0]:.2f},{mode_probs[1]:.2f},{mode_probs[2]:.2f}]")


if __name__ == "__main__":
    main()
