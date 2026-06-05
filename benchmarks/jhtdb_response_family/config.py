from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import tau
from pathlib import Path


TESTING_AUTH_TOKEN = "edu.jhu.pha.turbulence.testing-201406"


@dataclass(frozen=True)
class JHTDBAccessConfig:
    """Access and cache settings for a JHTDB point-sampling client.

    The default source is a deterministic local smoke field so CI and notebooks
    can execute without a network token. Set ``source_mode="pyjhtdb"`` for a
    real JHTDB run.
    """

    dataset: str = "isotropic1024coarse"
    source_mode: str = "synthetic"
    auth_token: str | None = None
    velocity_sinterp: str | int = "Lag4"
    gradient_sinterp: str | int = "FD4Lag4"
    tinterp: str | int = "NoTInt"
    cache_dir: Path = Path("benchmarks/jhtdb_response_family/cache")
    domain_length: float = tau
    max_points_per_request: int = 4096

    @property
    def token_source(self) -> str:
        """Identify where the auth token came from: public_testing | env_user_token | none."""
        if self.auth_token is not None:
            if self.auth_token == TESTING_AUTH_TOKEN:
                return "public_testing"
            return "env_user_token"
        if os.environ.get("JHTDB_TOKEN"):
            return "env_user_token"
        if self.source_mode == "pyjhtdb":
            return "public_testing"  # falls back to TESTING_AUTH_TOKEN
        return "none"

    @property
    def empirical(self) -> bool:
        """True only when using live pyjhtdb data."""
        return self.source_mode == "pyjhtdb"


@dataclass(frozen=True)
class JHTDBResponseFamilyConfig:
    """Small response-family benchmark configuration.

    Clustering settings are treated as frozen on the training time range before
    nearest-family retrieval is evaluated on the test time range.
    """

    access: JHTDBAccessConfig = field(default_factory=JHTDBAccessConfig)
    train_times: tuple[float, ...] = (0.0, 0.08)
    test_times: tuple[float, ...] = (0.16, 0.24)
    samples_per_time: int = 8
    seed: int = 0
    patch_points_per_axis: int = 3
    patch_spacing: float = tau / 1024.0 * 4.0
    future_horizon: float = 0.04
    response_quantity: str = "enstrophy_signed_delta"
    n_response_families: int = 3
    clustering_method: str = "kmeans"
    family_train_split: str = "train"
    eval_split: str = "test"
    neighbor_ks: tuple[int, ...] = (5, 10)
    feature_spaces: tuple[str, ...] = (
        "random",
        "vorticity_magnitude",
        "enstrophy",
        "kinetic_energy",
        "strain_magnitude",
        "q_criterion",
        "q_r_invariants",
        "strain_vorticity_alignment",
        "full_tensor_organization",
        "ucp_organization_k",
    )
    standardize_euclidean_features: bool = True

    @property
    def patch_size(self) -> int:
        return int(self.patch_points_per_axis)

