import numpy as np
from dataclasses import dataclass, field

@dataclass
class DynamicsConfig:
    # Drift coefficients
    alpha: float = 0.1
    beta: float = 0.05
    gamma: float = 0.1
    delta: float = 0.05
    eta: float = 0.05
    lam: float = 0.2

    # Noise
    noise_std: tuple = (0.01, 0.01, 0.01)

    # Event probabilities
    p_insight: float = 0.05
    p_backtrack: float = 0.05

    # Event magnitudes
    insight_delta: np.ndarray = field(
        default_factory=lambda: np.array([0.15, 0.10, -0.15])
        )
    backtrack_delta: np.ndarray = field(
        default_factory=lambda: np.array([-0.15, -0.10, 0.15])
    )

    clip_state: bool = True

class ReasoningDynamics:
    """
    Latent reasoning dynamics with smooth drift and rare discrete events

    State x = [progress, coherence, uncertainty]
    """

    def __init__(self, config: DynamicsConfig | None = None):
        self.cfg = config if config is not None else DynamicsConfig()

    def drift(self, x):
        """Smooth deterministic drift g(x)"""
        p, c, u = x

        dp = self.cfg.alpha * c - self.cfg.beta * u
        dc = self.cfg.gamma * (1.0 - u) - self.cfg.delta * c
        du = self.cfg.eta * (1.0 - c) - self.cfg.lam * p

        return np.array([dp, dc, du])

    def sample_event(self):
        """Sample a discrete reasoning event"""
        r = np.random.rand()

        if r < self.cfg.p_insight:
            return self.cfg.insight_delta
        elif r < self.cfg.p_insight + self.cfg.p_backtrack:
            return self.cfg.backtrack_delta
        else:
            return np.zeros(3)

    def step(self, x):
        # Smooth drift
        x_next = x + self.drift(x)

        # Gaussian process noise
        noise = np.random.randn(3) * np.array(self.cfg.noise_std)
        x_next = x_next + noise

        # Rare discrete event
        x_next = x_next + self.sample_event()

        # Clipping
        if self.cfg.clip_state:
            x_next = np.clip(x_next, 0.0, 1.0)

        return x_next
