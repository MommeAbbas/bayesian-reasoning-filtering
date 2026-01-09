import numpy as np
from src.simulation.slds import SLDSConfig, SLDSDynamics, SLDSSimulator, Mode


def main():
    # Deterministic-ish sanity test: reduce noise so you can see mode effects
    cfg = SLDSConfig(noise_std=(0.0, 0.0, 0.0))
    dyn = SLDSDynamics(cfg)
    sim = SLDSSimulator(dyn)

    xs, zs = sim.run(T=30, x0=np.array([0.2, 0.5, 0.8]), z0=Mode.NORMAL)

    print("k  mode   p      c      u")
    for k in range(xs.shape[0]):
        mode = Mode(int(zs[k])).name
        p, c, u = xs[k]
        print(f"{k:02d} {mode:9s} {p:.3f}  {c:.3f}  {u:.3f}")


if __name__ == "__main__":
    main()