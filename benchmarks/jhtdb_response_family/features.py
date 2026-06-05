from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


SCALAR_FEATURE_SPACES: dict[str, tuple[str, ...]] = {
    "scalar": ("vorticity_magnitude", "enstrophy", "kinetic_energy", "strain_magnitude"),
    "vorticity_magnitude": ("vorticity_magnitude",),
    "enstrophy": ("enstrophy",),
    "kinetic_energy": ("kinetic_energy",),
    "strain_magnitude": ("strain_magnitude",),
    "q_criterion": ("q_criterion",),
    "q_r": ("q_criterion", "r_invariant"),
    "q_r_invariants": ("q_criterion", "r_invariant"),
}

VECTOR_FEATURE_COLUMNS: dict[str, str] = {
    "random": "random_feature_vector",
    "alignment": "strain_vorticity_alignment_vector",
    "strain_vorticity_alignment": "strain_vorticity_alignment_vector",
    "oriented_transport": "oriented_transport_distribution",
    "oriented_transport_k": "oriented_transport_distribution",
    "patch_enstrophy_distribution": "pre_enstrophy_distribution",
    "patch_enstrophy_k": "pre_enstrophy_distribution",
    "strain_eigen_transport": "strain_eigenbasis_transport_distribution",
    "strain_eigenbasis_transport": "strain_eigenbasis_transport_distribution",
    "tensor": "tensor_organization_vector",
    "full_tensor_organization": "tensor_organization_vector",
    "K": "ucp_organization_distribution",
    "ucp_organization_k": "ucp_organization_distribution",
}

PRE_ORACLE_FEATURE_SPACES = tuple(SCALAR_FEATURE_SPACES) + tuple(VECTOR_FEATURE_COLUMNS)


def compute_patch_features(
    velocity: np.ndarray,
    gradient: np.ndarray,
    *,
    center_index: int,
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
    velocity = np.asarray(velocity, dtype=float).reshape((-1, 3))
    gradient = np.asarray(gradient, dtype=float).reshape((-1, 3, 3))
    center_velocity = velocity[center_index]
    center_gradient = gradient[center_index]

    strain, rotation = strain_rotation(center_gradient)
    omega = vorticity_from_gradient(center_gradient)
    omega_norm = _norm(omega)
    omega_dir = _unit(omega)
    strain_evals, strain_evecs = np.linalg.eigh(strain)
    order = np.argsort(strain_evals)
    strain_evals = strain_evals[order]
    strain_evecs = strain_evecs[:, order]
    alignment = np.abs(strain_evecs.T @ omega_dir) if omega_norm > 1e-12 else np.zeros(3)

    strain_norm = _norm(strain)
    rotation_norm = _norm(rotation)
    gradient_norm = _norm(center_gradient)
    q_criterion = 0.5 * (rotation_norm**2 - strain_norm**2)
    r_invariant = -float(np.linalg.det(center_gradient))
    gradient_scale2 = max(gradient_norm**2, 1e-12)
    gradient_scale3 = max(gradient_norm**3, 1e-12)

    patch_omega = vorticity_from_gradient(gradient)
    patch_enstrophy = np.sum(patch_omega * patch_omega, axis=1)
    pre_enstrophy_distribution = _normalize_nonnegative(patch_enstrophy)

    normalized_gradient = center_gradient / max(gradient_norm, 1e-12)
    normalized_strain_evals = strain_evals / max(strain_norm, 1e-12)
    tensor_organization = np.concatenate(
        [
            normalized_gradient.ravel(),
            normalized_strain_evals,
            omega_dir,
            alignment,
            np.asarray([q_criterion / gradient_scale2, r_invariant / gradient_scale3], dtype=float),
        ]
    )
    q_r_positive_parts = np.asarray(
        [
            max(q_criterion, 0.0) / gradient_scale2,
            max(-q_criterion, 0.0) / gradient_scale2,
            max(r_invariant, 0.0) / gradient_scale3,
            max(-r_invariant, 0.0) / gradient_scale3,
        ],
        dtype=float,
    )
    organization_distribution = _normalize_nonnegative(
        np.concatenate(
            [
                np.abs(normalized_strain_evals),
                np.abs(omega_dir),
                alignment,
                q_r_positive_parts,
                pre_enstrophy_distribution,
            ]
        )
    )
    strain_eigenbasis_transport = strain_eigenbasis_transport_distribution(gradient)
    oriented_transport = oriented_transport_distribution(gradient)

    return {
        "cheap_feature_time": "t",
        "feature_provenance": "pre_oracle_t_only",
        "velocity_vector": center_velocity.tolist(),
        "velocity_gradient_tensor": center_gradient.ravel().tolist(),
        "normalized_velocity_gradient_tensor": normalized_gradient.ravel().tolist(),
        "strain_tensor": strain.ravel().tolist(),
        "rotation_tensor": rotation.ravel().tolist(),
        "vorticity_vector": omega.tolist(),
        "vorticity_direction": omega_dir.tolist(),
        "strain_eigenvalues": strain_evals.tolist(),
        "strain_vorticity_alignment_vector": alignment.tolist(),
        "tensor_organization_vector": tensor_organization.tolist(),
        "ucp_organization_distribution": organization_distribution.tolist(),
        "oriented_transport_distribution": oriented_transport.tolist(),
        "strain_eigenbasis_transport_distribution": strain_eigenbasis_transport.tolist(),
        "pre_enstrophy_distribution": pre_enstrophy_distribution.tolist(),
        "random_feature_vector": random_feature_vector(scenario_id, seed).tolist(),
        "vorticity_magnitude": float(omega_norm),
        "enstrophy": float(omega_norm**2),
        "kinetic_energy": float(0.5 * np.dot(center_velocity, center_velocity)),
        "strain_magnitude": float(np.sqrt(2.0) * strain_norm),
        "q_criterion": float(q_criterion),
        "r_invariant": float(r_invariant),
    }


def strain_eigenbasis_transport_distribution(gradient: np.ndarray) -> np.ndarray:
    """Rotation-quotiented patch organization distribution.

    Bins are ordered as radial shell x strain eigen-axis x vorticity-alignment
    bucket. Mass is local enstrophy weighted by strain-mode strength and
    vorticity projection onto the local strain eigenbasis. The final vector is a
    shape-only distribution, so total intensity is intentionally quotiented out.
    """

    gradient = np.asarray(gradient, dtype=float).reshape((-1, 3, 3))
    n = len(gradient)
    axis_count = int(round(n ** (1.0 / 3.0)))
    if axis_count**3 != n or axis_count < 1:
        axis_count = 1
    center = (axis_count - 1) / 2.0
    max_radius = max(center, 1.0)
    distribution = np.zeros(27, dtype=float)

    for idx, local_gradient in enumerate(gradient):
        shell = _radial_shell(idx, axis_count, center, max_radius)
        local_strain, local_rotation = strain_rotation(local_gradient)
        omega = vorticity_from_gradient(local_gradient)
        enstrophy = float(np.dot(omega, omega))
        if enstrophy <= 1e-12:
            continue

        evals, evecs = np.linalg.eigh(local_strain)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
        strain_weights = _normalize_nonnegative(np.abs(evals))
        if not np.any(strain_weights):
            strain_weights = np.full(3, 1.0 / 3.0)

        omega_dir = _unit(omega)
        alignments = np.abs(evecs.T @ omega_dir)
        alignment_total = float(np.sum(alignments))
        if alignment_total <= 1e-12:
            alignments = np.full(3, 1.0 / 3.0)
        else:
            alignments = alignments / alignment_total

        # Rotation dominance is folded in as a gentle gate, not a new axis, so
        # the support stays small and transport distances stay interpretable.
        strain_norm = _norm(local_strain)
        rotation_norm = _norm(local_rotation)
        balance = rotation_norm / max(strain_norm + rotation_norm, 1e-12)
        balance_weight = 0.5 + balance

        for eigen_axis in range(3):
            alignment_bucket = min(2, int(np.floor(np.clip(abs(evecs[:, eigen_axis] @ omega_dir), 0.0, 1.0) * 3.0)))
            bin_index = shell * 9 + eigen_axis * 3 + alignment_bucket
            distribution[bin_index] += enstrophy * strain_weights[eigen_axis] * alignments[eigen_axis] * balance_weight

    return _normalize_nonnegative(distribution)


def oriented_transport_distribution(gradient: np.ndarray) -> np.ndarray:
    """Orientation-preserving patch organization distribution.

    Bins are ordered as radial shell x relative strain bin x local Q/R balance
    bin x signed global vorticity direction bin x strain-vorticity alignment
    bin. Unlike the eigenbasis quotient, this keeps global orientation as a
    fiber-like coordinate instead of rotating it away.
    """

    gradient = np.asarray(gradient, dtype=float).reshape((-1, 3, 3))
    n = len(gradient)
    axis_count = int(round(n ** (1.0 / 3.0)))
    if axis_count**3 != n or axis_count < 1:
        axis_count = 1
    center = (axis_count - 1) / 2.0
    max_radius = max(center, 1.0)

    strain_norms = np.zeros(n, dtype=float)
    local_parts: list[tuple[np.ndarray, np.ndarray, float, float, float]] = []
    for local_gradient in gradient:
        local_strain, local_rotation = strain_rotation(local_gradient)
        omega = vorticity_from_gradient(local_gradient)
        strain_norm = _norm(local_strain)
        rotation_norm = _norm(local_rotation)
        strain_norms[len(local_parts)] = strain_norm
        local_parts.append((local_strain, local_rotation, strain_norm, rotation_norm, float(np.dot(omega, omega))))

    strain_reference = float(np.median(strain_norms[strain_norms > 1e-12])) if np.any(strain_norms > 1e-12) else 1.0
    distribution = np.zeros(486, dtype=float)

    for idx, local_gradient in enumerate(gradient):
        local_strain, _local_rotation, strain_norm, rotation_norm, enstrophy = local_parts[idx]
        if enstrophy <= 1e-12:
            continue
        omega = vorticity_from_gradient(local_gradient)
        omega_norm = _norm(omega)
        if omega_norm <= 1e-12:
            continue
        omega_dir = omega / omega_norm

        shell = _radial_shell(idx, axis_count, center, max_radius)
        strain_bin = _three_bin(strain_norm / max(strain_reference, 1e-12), 0.75, 1.25)
        q_balance = (rotation_norm**2 - strain_norm**2) / max(rotation_norm**2 + strain_norm**2, 1e-12)
        q_bin = _three_bin(q_balance, -1.0 / 3.0, 1.0 / 3.0)
        orientation_bin = _signed_axis_bin(omega_dir)

        evals, evecs = np.linalg.eigh(local_strain)
        order = np.argsort(evals)
        evecs = evecs[:, order]
        max_alignment = float(np.max(np.abs(evecs.T @ omega_dir))) if strain_norm > 1e-12 else 0.0
        alignment_bin = _three_bin(max_alignment, 0.70, 0.86)

        bin_index = (((shell * 3 + strain_bin) * 3 + q_bin) * 6 + orientation_bin) * 3 + alignment_bin
        distribution[bin_index] += enstrophy

    return _normalize_nonnegative(distribution)


def strain_rotation(gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gradient = np.asarray(gradient, dtype=float).reshape(3, 3)
    strain = 0.5 * (gradient + gradient.T)
    rotation = 0.5 * (gradient - gradient.T)
    return strain, rotation


def vorticity_from_gradient(gradient: np.ndarray) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float)
    return np.stack(
        [
            gradient[..., 2, 1] - gradient[..., 1, 2],
            gradient[..., 0, 2] - gradient[..., 2, 0],
            gradient[..., 1, 0] - gradient[..., 0, 1],
        ],
        axis=-1,
    )


def random_feature_vector(scenario_id: str, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}|{scenario_id}".encode("utf-8")).digest()
    values = np.frombuffer(digest[:24], dtype=np.uint64).astype(float)
    return values / float(np.iinfo(np.uint64).max)


def _radial_shell(idx: int, axis_count: int, center: float, max_radius: float) -> int:
    if axis_count <= 1:
        return 0
    x = idx // (axis_count * axis_count)
    y = (idx // axis_count) % axis_count
    z = idx % axis_count
    radius = float(np.linalg.norm(np.asarray([x - center, y - center, z - center], dtype=float)))
    normalized = radius / max(max_radius, 1e-12)
    return min(2, int(np.floor(np.clip(normalized, 0.0, 0.999999) * 3.0)))


def _three_bin(value: float, low: float, high: float) -> int:
    if value < low:
        return 0
    if value < high:
        return 1
    return 2


def _signed_axis_bin(direction: np.ndarray) -> int:
    direction = np.asarray(direction, dtype=float).ravel()
    if direction.size != 3 or _norm(direction) <= 1e-12:
        return 0
    axis = int(np.argmax(np.abs(direction)))
    sign_offset = 0 if direction[axis] >= 0.0 else 1
    return axis * 2 + sign_offset


def array_payload(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=float)
    if isinstance(value, str):
        import ast

        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return np.zeros(0, dtype=float)
    if not isinstance(value, (list, tuple, np.ndarray)):
        return np.zeros(0, dtype=float)
    return np.asarray(value, dtype=float).ravel()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = _norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float)))


def _normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(np.sum(values))
    if total <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return values / total
