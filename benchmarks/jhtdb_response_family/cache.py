from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import JHTDBAccessConfig
from .jhtdb_client import FlowPointClient, PointData


def cached_query_points(
    client: FlowPointClient,
    config: JHTDBAccessConfig,
    *,
    time: float,
    points: np.ndarray,
    role: str,
) -> PointData:
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(
        dataset=config.dataset,
        source=client.source_name,
        time=time,
        points=points,
        role=role,
    )
    path = cache_dir / f"{key}.npz"
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        return PointData(
            velocity=np.asarray(loaded["velocity"], dtype=float),
            gradient=np.asarray(loaded["gradient"], dtype=float),
            dataset=str(loaded["dataset"]),
            time=float(loaded["time"]),
            source=str(loaded["source"]),
        )

    data = _query_points_chunked(client, config, time=time, points=points)
    np.savez_compressed(
        path,
        velocity=data.velocity,
        gradient=data.gradient,
        points=np.asarray(points, dtype=float),
        dataset=np.asarray(data.dataset),
        time=np.asarray(data.time),
        source=np.asarray(data.source),
        role=np.asarray(role),
    )
    return data


def _query_points_chunked(
    client: FlowPointClient,
    config: JHTDBAccessConfig,
    *,
    time: float,
    points: np.ndarray,
) -> PointData:
    points = np.asarray(points, dtype=float)
    max_points = max(1, int(config.max_points_per_request))
    if len(points) <= max_points:
        return client.query_points(config.dataset, time, points)

    chunks: list[PointData] = []
    for start in range(0, len(points), max_points):
        chunks.append(client.query_points(config.dataset, time, points[start : start + max_points]))
    return PointData(
        velocity=np.concatenate([chunk.velocity for chunk in chunks], axis=0),
        gradient=np.concatenate([chunk.gradient for chunk in chunks], axis=0),
        dataset=chunks[0].dataset,
        time=float(chunks[0].time),
        source=chunks[0].source,
    )


def cache_key(*, dataset: str, source: str, time: float, points: np.ndarray, role: str) -> str:
    rounded = np.round(np.asarray(points, dtype=float), decimals=12)
    digest = hashlib.sha256()
    digest.update(dataset.encode("utf-8"))
    digest.update(source.encode("utf-8"))
    digest.update(role.encode("utf-8"))
    digest.update(f"{float(time):.12g}".encode("utf-8"))
    digest.update(rounded.tobytes())
    return digest.hexdigest()[:24]
