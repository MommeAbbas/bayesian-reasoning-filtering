import numpy as np
from src.simulation.dynamics import ReasoningDynamics, DynamicsConfig


def main():
    # Config with noise turned OFF for deterministic testing
    cfg = DynamicsConfig(
        noise_std=(0.0, 0.0, 0.0),
        p_insight=0.0,
        p_backtrack=0.0,
    )

    dyn = ReasoningDynamics(config=cfg)

    # Initial latent reasoning state: [progress, coherence, uncertainty]
    x = np.array([0.2, 0.5, 0.8])

    print("Initial state:", x)

    for k in range(10):
        x = dyn.step(x)
        print(f"Step {k + 1}: {x}")


if __name__ == "__main__":
    main()