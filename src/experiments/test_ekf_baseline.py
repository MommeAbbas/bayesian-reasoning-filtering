import numpy as np

from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator, Mode
from src.simulation.sensors import SensorConfig, ReasoningSensors
from src.filters.ekf_baseline import EKFBaseline, EKFConfig


def main():
    np.random.seed(0)

    dyn = SLDSDynamics(SLDSConfig())
    sensors = ReasoningSensors(SensorConfig(noise_std=(0.05, 0.05, 0.05), outlier_prob=0.05, outlier_scale=25.0))

    xs, zs = SLDSSimulator(dyn).run(T=25, x0=np.array([0.2, 0.5, 0.8]), z0=Mode.NORMAL)

    ekf = EKFBaseline(dyn=dyn, sensors=sensors, cfg=EKFConfig())

    print("k  true_mode  true_p  ekf_p")
    for k in range(1, 26):
        y = sensors.observe(xs[k])
        mu, Sigma = ekf.step(y)
        print(f"{k:02d} {Mode(int(zs[k])).name:9s} {xs[k,0]:.3f}  {mu[0]:.3f}")


if __name__ == "__main__":
    main()
