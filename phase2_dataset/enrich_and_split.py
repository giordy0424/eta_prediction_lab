#!/usr/bin/env python3
"""
Script 2: Arricchimento OSRM + Train/Test split.
Merge di osrm_enrich_dataset.py e dataset_splitter.py.

Input : CSV da prepare_dataset.py (con coordinate originali + trip_duration + feature temporali)
Output: ultimate_train.csv + ultimate_test.csv

Nessun dataset intermedio salvato (elimina il file temporaneo in input).
"""

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

OSRM_ROUTE_PATH = "/route/v1/driving"
DEFAULT_OSRM_URL = "http://localhost:5000"
SOGLIA_INCROCI_COMPLESSI = 4
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 3.0

OUTPUT_FIELDS = [
    "origin_lat", "origin_lon", "destination_lat", "destination_lon",
    "osrm_eta", "osrm_distance", "svolte_totali", "svolte_sinistra",
    "svolte_destra", "rotonde", "semafori_totali", "incroci_totali",
    "incroci_complessi", "nodi_attraversati", "trip_duration",
    "pickup_hour", "pickup_dayofweek", "pickup_month",
]


def _count_turns(steps: List[Dict]) -> Tuple[int, int, int]:
    left_modifiers = {"left", "sharp left", "slight left"}
    right_modifiers = {"right", "sharp right", "slight right"}
    turn_types = {"turn", "end of road", "fork", "on ramp", "off ramp", "new name"}
    total, sinistra, destra = 0, 0, 0
    for step in steps:
        maneuver = step.get("maneuver", {})
        m_type = maneuver.get("type", "")
        modifier = maneuver.get("modifier", "")
        if m_type in turn_types and modifier:
            total += 1
            if modifier in left_modifiers:
                sinistra += 1
            elif modifier in right_modifiers:
                destra += 1
    return total, sinistra, destra


def _count_roundabouts(steps: List[Dict]) -> int:
    return sum(1 for step in steps if step.get("maneuver", {}).get("type", "") in ("roundabout", "rotary"))


def _count_intersections_and_signals(steps: List[Dict]) -> Tuple[int, int, int, int]:
    total, complessi, semafori, nodi = 0, 0, 0, 0
    for step in steps:
        intersections = step.get("intersections", [])
        for ix in intersections:
            total += 1
            bearings = ix.get("bearings", [])
            if len(bearings) >= SOGLIA_INCROCI_COMPLESSI:
                complessi += 1
            if ix.get("traffic_signal", False):
                semafori += 1
            classes = ix.get("classes", [])
            if isinstance(classes, list) and "traffic_signal" in classes:
                semafori += 1
        geom = step.get("geometry", "")
        if isinstance(geom, dict):
            coords = geom.get("coordinates", [])
            nodi += len(coords)
        elif isinstance(geom, str) and len(geom) > 0:
            try:
                import polyline as pl
                decoded = pl.decode(geom)
                nodi += len(decoded)
            except ImportError:
                dist = step.get("distance", 0)
                nodi += max(1, int(dist / 10))
    return total, complessi, semafori, nodi


def parse_osrm_response(data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "origin_lat": None, "origin_lon": None,
        "destination_lat": None, "destination_lon": None,
        "osrm_eta": None, "osrm_distance": None,
        "svolte_totali": None, "svolte_sinistra": None,
        "svolte_destra": None, "rotonde": None,
        "semafori_totali": None, "incroci_totali": None,
        "incroci_complessi": None, "nodi_attraversati": None,
    }
    routes = data.get("routes", [])
    if not routes:
        return result
    route = routes[0]
    waypoints = data.get("waypoints", [])
    if len(waypoints) >= 2:
        origin = waypoints[0].get("location", [])
        dest = waypoints[-1].get("location", [])
        if len(origin) >= 2:
            result["origin_lon"] = origin[0]
            result["origin_lat"] = origin[1]
        if len(dest) >= 2:
            result["destination_lon"] = dest[0]
            result["destination_lat"] = dest[1]
    result["osrm_eta"] = route.get("duration")
    result["osrm_distance"] = route.get("distance")
    legs = route.get("legs", [])
    all_steps: List[Dict] = []
    for leg in legs:
        all_steps.extend(leg.get("steps", []))
    if all_steps:
        st, sl, sr = _count_turns(all_steps)
        result["svolte_totali"] = st
        result["svolte_sinistra"] = sl
        result["svolte_destra"] = sr
        result["rotonde"] = _count_roundabouts(all_steps)
        it, ic, sem, nodi = _count_intersections_and_signals(all_steps)
        result["incroci_totali"] = it
        result["incroci_complessi"] = ic
        result["semafori_totali"] = sem
        result["nodi_attraversati"] = nodi
    return result


class FailedRequestLogger:
    def __init__(self, log_path: Optional[str]):
        self._enabled = log_path is not None
        self._lock = threading.Lock()
        if self._enabled:
            self._file = open(log_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=[
                "timestamp", "pickup_lon", "pickup_lat",
                "dropoff_lon", "dropoff_lat",
                "error_type", "error_detail", "attempt", "url",
            ])
            self._writer.writeheader()
            self._file.flush()

    def log_failure(self, pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
                    error_type, error_detail, attempt, url=""):
        if not self._enabled:
            return
        with self._lock:
            self._writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "pickup_lon": pickup_lon, "pickup_lat": pickup_lat,
                "dropoff_lon": dropoff_lon, "dropoff_lat": dropoff_lat,
                "error_type": error_type, "error_detail": error_detail,
                "attempt": attempt, "url": url,
            })
            self._file.flush()

    def close(self):
        if self._enabled:
            self._file.close()


def query_osrm(session, osrm_url, pickup_lon, pickup_lat,
               dropoff_lon, dropoff_lat, max_retries=DEFAULT_MAX_RETRIES,
               retry_backoff=DEFAULT_RETRY_BACKOFF,
               request_timeout=DEFAULT_REQUEST_TIMEOUT,
               failed_logger=None):
    coord_str = f"{pickup_lon},{pickup_lat};{dropoff_lon},{dropoff_lat}"
    url = f"{osrm_url.rstrip('/')}{OSRM_ROUTE_PATH}/{coord_str}"
    params = {"overview": "full", "steps": "true", "geometries": "geojson", "annotations": "true"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=request_timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "Ok":
                if failed_logger:
                    failed_logger.log_failure(
                        pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
                        "osrm_bad_code", f"OSRM code: {data.get('code')}", attempt, url)
                return None
            return parse_osrm_response(data)
        except requests.exceptions.Timeout:
            last_error = "timeout"
            detail = f"Richiesta scaduta dopo {request_timeout}s"
        except requests.exceptions.ConnectionError as e:
            last_error = "connection_error"
            detail = str(e)[:200]
        except requests.exceptions.HTTPError as e:
            last_error = "http_error"
            status = e.response.status_code if e.response is not None else "N/A"
            detail = f"HTTP {status}"
        except (json.JSONDecodeError, ValueError) as e:
            last_error = "json_decode_error"
            detail = str(e)[:200]
            if failed_logger:
                failed_logger.log_failure(
                    pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
                    last_error, detail, attempt, url)
            return None
        if failed_logger:
            failed_logger.log_failure(
                pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
                last_error, detail, attempt, url)
        if attempt < max_retries:
            wait = retry_backoff * (2 ** (attempt - 1))
            time.sleep(wait)
    return None


def _empty_osrm_data():
    return {
        "origin_lat": None, "origin_lon": None,
        "destination_lat": None, "destination_lon": None,
        "osrm_eta": None, "osrm_distance": None,
        "svolte_totali": None, "svolte_sinistra": None,
        "svolte_destra": None, "rotonde": None,
        "semafori_totali": None, "incroci_totali": None,
        "incroci_complessi": None, "nodi_attraversati": None,
    }


def process_row(row, session, osrm_url, max_retries, retry_backoff, request_timeout, failed_logger):
    try:
        pickup_lon = float(row["pickup_longitude"])
        pickup_lat = float(row["pickup_latitude"])
        dropoff_lon = float(row["dropoff_longitude"])
        dropoff_lat = float(row["dropoff_latitude"])
    except (ValueError, KeyError) as e:
        osrm_data = _empty_osrm_data()
    else:
        result = query_osrm(session, osrm_url, pickup_lon, pickup_lat,
                            dropoff_lon, dropoff_lat, max_retries,
                            retry_backoff, request_timeout, failed_logger)
        osrm_data = result if result is not None else _empty_osrm_data()
    return {
        **osrm_data,
        "trip_duration": row.get("trip_duration", ""),
        "pickup_hour": row.get("pickup_hour", ""),
        "pickup_dayofweek": row.get("pickup_dayofweek", ""),
        "pickup_month": row.get("pickup_month", ""),
    }


def run_pipeline(input_path, output_train, output_test, osrm_url=DEFAULT_OSRM_URL,
                 workers=8, batch_size=100, limit=None, skip_errors=True,
                 max_retries=DEFAULT_MAX_RETRIES, retry_backoff=DEFAULT_RETRY_BACKOFF,
                 request_timeout=DEFAULT_REQUEST_TIMEOUT, error_log=None, test_size=0.2):
    logger.info(f"Lettura dati da {input_path} ...")
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    total = len(rows)
    logger.info(f"Lette {total} righe")

    # File temporaneo per l'enriched data
    enriched_path = input_path + ".enriched.tmp"
    out_file = open(enriched_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
    writer.writeheader()
    out_file.flush()

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    failed_logger = FailedRequestLogger(error_log)
    completed = 0
    errors = 0

    def _write_result(enriched):
        nonlocal completed, errors
        if enriched["osrm_eta"] is None:
            errors += 1
        writer.writerow(enriched)
        completed += 1
        if completed % batch_size == 0:
            out_file.flush()
            logger.info(f"Progresso: {completed}/{total} ({100*completed/total:.1f}%) — errori: {errors}")

    logger.info(f"Avvio enrichment con {workers} worker su OSRM {osrm_url} ...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_row, row, session, osrm_url,
                          max_retries, retry_backoff, request_timeout, failed_logger): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                enriched = future.result()
                _write_result(enriched)
            except Exception as exc:
                errors += 1
                if skip_errors:
                    row = rows[idx]
                    empty = {**_empty_osrm_data(),
                             "trip_duration": row.get("trip_duration", ""),
                             "pickup_hour": row.get("pickup_hour", ""),
                             "pickup_dayofweek": row.get("pickup_dayofweek", ""),
                             "pickup_month": row.get("pickup_month", "")}
                    _write_result(empty)
                else:
                    raise

    out_file.close()
    failed_logger.close()
    logger.info(f"Enrichment completato: {completed} righe, {errors} errori")

    # Train/test split
    logger.info(f"Caricamento dati arricchiti per split ...")
    df = pd.read_csv(enriched_path)
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)

    df_train.to_csv(output_train, index=False)
    df_test.to_csv(output_test, index=False)
    logger.info(f"Train: {len(df_train)} righe → {output_train}")
    logger.info(f"Test:  {len(df_test)} righe → {output_test}")

    os.remove(enriched_path)
    os.remove(input_path)
    logger.info(f"File temporanei eliminati")


def main():
    parser = argparse.ArgumentParser(
        description="Arricchimento OSRM + Train/Test split"
    )
    parser.add_argument("--input", "-i", required=True, help="Input CSV (output di prepare_dataset.py)")
    parser.add_argument("--output-train", required=True, help="Output: ultimate_train.csv")
    parser.add_argument("--output-test", required=True, help="Output: ultimate_test.csv")
    parser.add_argument("--osrm-url", default=DEFAULT_OSRM_URL)
    parser.add_argument("--workers", "-w", type=int, default=8)
    parser.add_argument("--batch-size", "-b", type=int, default=100)
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--no-skip-errors", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff", type=float, default=DEFAULT_RETRY_BACKOFF)
    parser.add_argument("--error-log", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"File input non trovato: {args.input}")
        sys.exit(1)

    try:
        resp = requests.get(f"{args.osrm_url.rstrip('/')}/", timeout=5)
        logger.info(f"Server OSRM raggiungibile a {args.osrm_url}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Impossibile connettersi a OSRM a {args.osrm_url}. Avviare phase1.")
        sys.exit(1)

    run_pipeline(
        input_path=args.input,
        output_train=args.output_train,
        output_test=args.output_test,
        osrm_url=args.osrm_url,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        skip_errors=not args.no_skip_errors,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        request_timeout=args.request_timeout,
        error_log=args.error_log,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
