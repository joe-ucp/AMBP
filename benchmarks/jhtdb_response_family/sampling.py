from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import JHTDBResponseFamilyConfig


@dataclass(frozen=True)
class PatchRecord:
    scenario_id: str
    split: str
    sample_index: int
    time: float
    future_time: float
    center: np.ndarray
    points: np.ndarray
    center_index: int


def sample_patch_records(config: JHTDBResponseFamilyConfig) -> list[PatchRecord]:
    rng = np.random.default_rng(config.seed)
    records: list[PatchRecord] = []
    sample_index = 0
    for split, times in (("train", config.train_times), ("test", config.test_times)):
        for time in times:
            for _ in range(config.samples_per_time):
                center = rng.uniform(0.0, config.access.domain_length, size=3)
                points, center_index = patch_points(
                    center,
                    patch_points_per_axis=config.patch_points_per_axis,
                    patch_spacing=config.patch_spacing,
                    domain_length=config.access.domain_length,
                )
                scenario_id = "|".join(
                    [
                        config.access.dataset,
                        split,
                        f"t={float(time):.8g}",
                        f"dt={float(config.future_horizon):.8g}",
                        f"patch={config.patch_points_per_axis}",
                        f"i={sample_index}",
                    ]
                )
                records.append(
                    PatchRecord(
                        scenario_id=scenario_id,
                        split=split,
                        sample_index=sample_index,
                        time=float(time),
                        future_time=float(time + config.future_horizon),
                        center=center,
                        points=points,
                        center_index=center_index,
                    )
                )
                sample_index += 1
    return records


def patch_points(
    center: np.ndarray,
    *,
    patch_points_per_axis: int,
    patch_spacing: float,
    domain_length: float,
) -> tuple[np.ndarray, int]:
    if patch_points_per_axis < 1 or patch_points_per_axis % 2 != 1:
        raise ValueError("patch_points_per_axis must be a positive odd integer")
    radius = patch_points_per_axis // 2
    offsets_1d = np.arange(-radius, radius + 1, dtype=float) * float(patch_spacing)
    offsets = np.asarray(np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")).reshape(3, -1).T
    points = (np.asarray(center, dtype=float).reshape(1, 3) + offsets) % float(domain_length)
    center_index = int(np.where(np.all(np.isclose(offsets, 0.0), axis=1))[0][0])
    return points, center_index


def records_to_dataframe(records: list[PatchRecord], config: JHTDBResponseFamilyConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "scenario_id": record.scenario_id,
                "dataset": config.access.dataset,
                "case_name": config.access.dataset,
                "split": record.split,
                "sample_index": record.sample_index,
                "t": record.time,
                "future_t": record.future_time,
                "delta_t": config.future_horizon,
                "patch_size": config.patch_points_per_axis,
                "patch_spacing": config.patch_spacing,
                "patch_point_count": len(record.points),
                "x": float(record.center[0]),
                "y": float(record.center[1]),
                "z": float(record.center[2]),
            }
        )
    return pd.DataFrame(rows)

