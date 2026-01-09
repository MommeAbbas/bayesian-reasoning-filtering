import numpy as np
from src.simulation.sensors import ReasoningSensors, SensorConfig


def main():
    cfg = SensorConfig(
        noise_std=(0.02, 0.02, 0.02),
        outlier_prob=0.2,        # frequent outliers for testing
        outlier_scale=25.0,      # big outliers
    )
    sensors = ReasoningSensors(cfg)

    x = np.array([0.6, 0.7, 0.3], dtype=float)
    mu = sensors.h(x)

    print("Latent state:", x)
    print("h(x):", mu)

    for k in range(10):
        y = sensors.observe(x)
        ll = sensors.log_likelihood(y, x)
        print(f"{k+1:02d} y={y}  loglik={ll:.3f}")


if __name__ == "__main__":
    main()
