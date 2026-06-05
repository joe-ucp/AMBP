from __future__ import annotations

import importlib.util
import os
import time as time_module
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

import numpy as np

from .config import JHTDBAccessConfig, TESTING_AUTH_TOKEN


@dataclass(frozen=True)
class PointData:
    velocity: np.ndarray
    gradient: np.ndarray
    dataset: str
    time: float
    source: str


@dataclass(frozen=True)
class PressureHessianData:
    hessian: np.ndarray
    dataset: str
    time: float
    source: str


class FlowPointClient(Protocol):
    source_name: str

    def query_points(self, dataset: str, time: float, points: np.ndarray) -> PointData:
        ...


class SyntheticJHTDBClient:
    """Deterministic divergence-free smoke field with a JHTDB-like interface."""

    source_name = "synthetic_smoke_field"

    def query_points(self, dataset: str, time: float, points: np.ndarray) -> PointData:
        points = _as_points(points)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        t = float(time)

        velocity = np.column_stack(
            [
                np.sin(z + 0.70 * t) + np.cos(y - 0.35 * t) + 0.20 * np.sin(2.0 * z + 0.15 * t),
                np.sin(x + 0.55 * t) + np.cos(z - 0.70 * t) + 0.20 * np.sin(2.0 * x + 0.20 * t),
                np.sin(y + 0.35 * t) + np.cos(x - 0.55 * t) + 0.20 * np.sin(2.0 * y + 0.25 * t),
            ]
        )

        gradient = np.zeros((len(points), 3, 3), dtype=float)
        gradient[:, 0, 1] = -np.sin(y - 0.35 * t)
        gradient[:, 0, 2] = np.cos(z + 0.70 * t) + 0.40 * np.cos(2.0 * z + 0.15 * t)
        gradient[:, 1, 0] = np.cos(x + 0.55 * t) + 0.40 * np.cos(2.0 * x + 0.20 * t)
        gradient[:, 1, 2] = -np.sin(z - 0.70 * t)
        gradient[:, 2, 0] = -np.sin(x - 0.55 * t)
        gradient[:, 2, 1] = np.cos(y + 0.35 * t) + 0.40 * np.cos(2.0 * y + 0.25 * t)

        return PointData(
            velocity=velocity,
            gradient=gradient,
            dataset=dataset,
            time=t,
            source=self.source_name,
        )

    def query_pressure_hessian(self, dataset: str, time: float, points: np.ndarray) -> PressureHessianData:
        points = _as_points(points)
        return PressureHessianData(
            hessian=np.zeros((len(points), 3, 3), dtype=float),
            dataset=dataset,
            time=float(time),
            source=self.source_name,
        )


class PyJHTDBPointClient:
    """Thin optional adapter around ``pyJHTDB.libJHTDB.getData``."""

    source_name = "jhtdb_pyjhtdb"

    def __init__(self, config: JHTDBAccessConfig):
        self.config = config
        self.auth_token = config.auth_token or os.environ.get("JHTDB_TOKEN") or TESTING_AUTH_TOKEN
        self._client = None
        self._pyjhtdb = None

    def query_points(self, dataset: str, time: float, points: np.ndarray) -> PointData:
        points = _as_points(points).astype(np.float32, copy=False)
        if len(points) > self.config.max_points_per_request:
            raise ValueError(
                f"JHTDB request has {len(points)} points; testing token requests should stay below "
                f"{self.config.max_points_per_request}"
            )
        client = self._ensure_client()
        velocity_sinterp = self._interp_code(self.config.velocity_sinterp)
        gradient_sinterp = self._interp_code(self.config.gradient_sinterp)
        tinterp = self._interp_code(self.config.tinterp)
        velocity = client.getData(
            float(time),
            points,
            data_set=dataset,
            sinterp=velocity_sinterp,
            tinterp=tinterp,
            getFunction="getVelocity",
        )
        gradient = client.getData(
            float(time),
            points,
            data_set=dataset,
            sinterp=gradient_sinterp,
            tinterp=tinterp,
            getFunction="getVelocityGradient",
        )
        velocity = np.asarray(velocity, dtype=float).reshape((-1, 3))
        gradient = np.asarray(gradient, dtype=float).reshape((-1, 9)).reshape((-1, 3, 3))
        return PointData(
            velocity=velocity,
            gradient=gradient,
            dataset=dataset,
            time=float(time),
            source=self.source_name,
        )

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "finalize"):
            self._client.finalize()
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if importlib.util.find_spec("pyJHTDB") is None:
            raise ImportError("pyJHTDB is not installed; use source_mode='synthetic' for a local smoke run")
        import pyJHTDB

        self._pyjhtdb = pyJHTDB
        client = pyJHTDB.libJHTDB()
        client.initialize()
        if hasattr(client, "add_token"):
            client.add_token(self.auth_token)
        self._client = client
        return client

    def _interp_code(self, value: str | int) -> int:
        if isinstance(value, int):
            return value
        self._ensure_client()
        try:
            from pyJHTDB.dbinfo import interpolation_code

            if value in interpolation_code:
                return int(interpolation_code[value])
        except Exception:
            pass
        fallback = {
            "NoSInt": 0,
            "Lag4": 4,
            "Lag6": 6,
            "Lag8": 8,
            "NoTInt": 0,
            "FD4NoInt": 40,
            "FD6NoInt": 60,
            "FD8NoInt": 80,
            "FD4Lag4": 44,
        }
        if value not in fallback:
            raise ValueError(f"unknown JHTDB interpolation option: {value!r}")
        return fallback[value]


class SoapJHTDBPointClient:
    """Direct JHTDB SOAP web-service point client.

    This path is intentionally tiny: it calls the documented ``GetVelocity`` and
    ``GetVelocityGradient`` operations and parses the returned vector/tensor
    payloads. It avoids local gSOAP build requirements while still using live
    JHTDB data.
    """

    source_name = "jhtdb_soap_webservice"

    def __init__(self, config: JHTDBAccessConfig):
        self.config = config
        self.auth_token = config.auth_token or os.environ.get("JHTDB_TOKEN") or TESTING_AUTH_TOKEN
        self.endpoint = "https://turbulence.pha.jhu.edu/service/turbulence.asmx"

    def query_points(self, dataset: str, time: float, points: np.ndarray) -> PointData:
        points = _as_points(points)
        if len(points) > self.config.max_points_per_request:
            raise ValueError(
                f"JHTDB request has {len(points)} points; testing token requests should stay below "
                f"{self.config.max_points_per_request}"
            )
        velocity = self._call_vector_operation(
            "GetVelocity",
            dataset=dataset,
            time=time,
            points=points,
            spatial_interpolation=_soap_spatial_interpolation(self.config.velocity_sinterp),
            result_tag="Vector3",
            fields=("x", "y", "z"),
        ).reshape((-1, 3))
        gradient = self._call_vector_operation(
            "GetVelocityGradient",
            dataset=dataset,
            time=time,
            points=points,
            spatial_interpolation=_soap_spatial_interpolation(self.config.gradient_sinterp),
            result_tag="VelocityGradient",
            fields=("duxdx", "duxdy", "duxdz", "duydx", "duydy", "duydz", "duzdx", "duzdy", "duzdz"),
        ).reshape((-1, 3, 3))
        return PointData(
            velocity=velocity,
            gradient=gradient,
            dataset=dataset,
            time=float(time),
            source=self.source_name,
        )

    def query_pressure_hessian(self, dataset: str, time: float, points: np.ndarray) -> PressureHessianData:
        points = _as_points(points)
        if len(points) > self.config.max_points_per_request:
            raise ValueError(
                f"JHTDB request has {len(points)} points; testing token requests should stay below "
                f"{self.config.max_points_per_request}"
            )
        packed = self._call_vector_operation(
            "GetPressureHessian",
            dataset=dataset,
            time=time,
            points=points,
            spatial_interpolation=_soap_spatial_interpolation(self.config.gradient_sinterp),
            result_tag="PressureHessian",
            fields=("d2pdxdx", "d2pdxdy", "d2pdxdz", "d2pdydy", "d2pdydz", "d2pdzdz"),
        )
        hessian = np.zeros((len(points), 3, 3), dtype=float)
        hessian[:, 0, 0] = packed[:, 0]
        hessian[:, 0, 1] = packed[:, 1]
        hessian[:, 1, 0] = packed[:, 1]
        hessian[:, 0, 2] = packed[:, 2]
        hessian[:, 2, 0] = packed[:, 2]
        hessian[:, 1, 1] = packed[:, 3]
        hessian[:, 1, 2] = packed[:, 4]
        hessian[:, 2, 1] = packed[:, 4]
        hessian[:, 2, 2] = packed[:, 5]
        return PressureHessianData(
            hessian=hessian,
            dataset=dataset,
            time=float(time),
            source=self.source_name,
        )

    def _call_vector_operation(
        self,
        method: str,
        *,
        dataset: str,
        time: float,
        points: np.ndarray,
        spatial_interpolation: str,
        result_tag: str,
        fields: tuple[str, ...],
    ) -> np.ndarray:
        body = self._soap_body(
            method,
            dataset=dataset,
            time=time,
            points=points,
            spatial_interpolation=spatial_interpolation,
        )
        req = request.Request(self.endpoint, data=body)
        req.add_header("Content-Type", "text/xml; charset=utf-8")
        req.add_header("SOAPAction", f'"http://turbulence.pha.jhu.edu/{method}"')
        payload = None
        last_error: Exception | None = None
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                with request.urlopen(req, timeout=180) as response:
                    payload = response.read()
                break
            except (TimeoutError, error.URLError, OSError) as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    raise
                time_module.sleep(min(30.0, 2.0 * (attempt + 1)))
        if payload is None:
            raise RuntimeError("JHTDB SOAP request failed without a response") from last_error
        return _parse_soap_vectors(payload, result_tag=result_tag, fields=fields)

    def _soap_body(
        self,
        method: str,
        *,
        dataset: str,
        time: float,
        points: np.ndarray,
        spatial_interpolation: str,
    ) -> bytes:
        points_xml = "".join(
            "<Point3>"
            f"<x>{float(point[0]):.12g}</x>"
            f"<y>{float(point[1]):.12g}</y>"
            f"<z>{float(point[2]):.12g}</z>"
            "</Point3>"
            for point in points
        )
        temporal = _soap_temporal_interpolation(self.config.tinterp)
        xml = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <{method} xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{escape(self.auth_token)}</authToken>
      <dataset>{escape(dataset)}</dataset>
      <time>{float(time):.12g}</time>
      <spatialInterpolation>{escape(spatial_interpolation)}</spatialInterpolation>
      <temporalInterpolation>{escape(temporal)}</temporalInterpolation>
      <points>{points_xml}</points>
      <addr></addr>
    </{method}>
  </soap:Body>
</soap:Envelope>"""
        return xml.encode("utf-8")


def make_flow_client(config: JHTDBAccessConfig) -> FlowPointClient:
    if config.source_mode == "synthetic":
        return SyntheticJHTDBClient()
    if config.source_mode == "soap":
        return SoapJHTDBPointClient(config)
    if config.source_mode == "pyjhtdb":
        if importlib.util.find_spec("pyJHTDB") is not None:
            client = PyJHTDBPointClient(config)
            try:
                client._ensure_client()
                return client
            except Exception:
                return SoapJHTDBPointClient(config)
        return SoapJHTDBPointClient(config)
    if config.source_mode == "auto":
        if os.environ.get("JHTDB_LIVE") == "1" and importlib.util.find_spec("pyJHTDB") is not None:
            return PyJHTDBPointClient(config)
        return SyntheticJHTDBClient()
    raise ValueError("source_mode must be 'synthetic', 'pyjhtdb', 'soap', or 'auto'")


def _as_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return array


def _soap_spatial_interpolation(value: str | int) -> str:
    if isinstance(value, int):
        reverse = {0: "None", 4: "Lag4", 6: "Lag6", 8: "Lag8", 40: "None_Fd4", 44: "Fd4Lag4"}
        return reverse.get(value, str(value))
    normalized = value.strip()
    aliases = {
        "none": "None",
        "nosint": "None",
        "lag4": "Lag4",
        "lag6": "Lag6",
        "lag8": "Lag8",
        "fd4lag4": "Fd4Lag4",
        "fd4noint": "None_Fd4",
        "none_fd4": "None_Fd4",
        "fd6noint": "None_Fd6",
        "none_fd6": "None_Fd6",
        "fd8noint": "None_Fd8",
        "none_fd8": "None_Fd8",
    }
    return aliases.get(normalized.lower(), normalized)


def _soap_temporal_interpolation(value: str | int) -> str:
    if isinstance(value, int):
        return "None" if value == 0 else "PCHIP"
    normalized = value.strip()
    aliases = {"none": "None", "notint": "None", "pchip": "PCHIP"}
    return aliases.get(normalized.lower(), normalized)


def _parse_soap_vectors(payload: bytes, *, result_tag: str, fields: tuple[str, ...]) -> np.ndarray:
    root = ET.fromstring(payload)
    rows: list[list[float]] = []
    for node in root.iter():
        if _local_name(node.tag) != result_tag:
            continue
        row: list[float] = []
        children = {_local_name(child.tag): child.text for child in list(node)}
        for field in fields:
            value = children.get(field)
            if value is None:
                raise ValueError(f"SOAP response missing field {field!r} in {result_tag}")
            row.append(float(value))
        rows.append(row)
    if not rows:
        text = payload.decode("utf-8", errors="replace")
        raise ValueError(f"SOAP response did not include {result_tag}: {text[:500]}")
    return np.asarray(rows, dtype=float)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
