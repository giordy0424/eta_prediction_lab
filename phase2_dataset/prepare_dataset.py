#!/usr/bin/env python3
"""
Script 1: Filtraggio spaziale + conversione datetime.
Merge di data_filtering.ipynb e datetime_converter.ipynb.

Input : train.csv (Kaggle NYC Taxi) + mask.geojson (poligono Manhattan)
Output: stdout o file temporaneo (dati filtrati con feature temporali)

Nessun dataset intermedio salvato.
"""

import sys
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filtraggio spaziale a Manhattan + feature temporali"
    )
    parser.add_argument("--input", "-i", required=True, help="Kaggle train.csv")
    parser.add_argument("--mask", "-m", required=True, help="mask.geojson (Manhattan polygon)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV (pass-through per enrich_and_split.py)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input non trovato: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.mask).exists():
        print(f"ERROR: Mask non trovata: {args.mask}", file=sys.stderr)
        sys.exit(1)

    # 1. Carica dati originali
    print(f"Caricamento {args.input} ...", file=sys.stderr)
    df_raw = pd.read_csv(args.input)
    print(f"Totale viaggi originali: {len(df_raw)}", file=sys.stderr)

    # 2. Carica maschera Manhattan
    mask = gpd.read_file(args.mask)
    crs_wgs84 = "EPSG:4326"
    if mask.crs != crs_wgs84:
        mask = mask.to_crs(crs_wgs84)

    # 3. Filtro spaziale: pickup AND dropoff within Manhattan
    gdf_pickup = gpd.GeoDataFrame(
        df_raw[["id", "pickup_longitude", "pickup_latitude"]],
        geometry=gpd.points_from_xy(df_raw["pickup_longitude"], df_raw["pickup_latitude"]),
        crs=crs_wgs84,
    )
    pickup_dentro = gpd.sjoin(gdf_pickup, mask, how="inner", predicate="within")
    id_origini_valide = set(pickup_dentro["id"])

    gdf_dropoff = gpd.GeoDataFrame(
        df_raw[["id", "dropoff_longitude", "dropoff_latitude"]],
        geometry=gpd.points_from_xy(df_raw["dropoff_longitude"], df_raw["dropoff_latitude"]),
        crs=crs_wgs84,
    )
    dropoff_dentro = gpd.sjoin(gdf_dropoff, mask, how="inner", predicate="within")
    id_destinazioni_valide = set(dropoff_dentro["id"])

    id_validi = id_origini_valide.intersection(id_destinazioni_valide)
    df = df_raw[df_raw["id"].isin(id_validi)].copy().reset_index(drop=True)
    print(f"Viaggi filtrati (pickup AND dropoff in Manhattan): {len(df)}", file=sys.stderr)

    # 4. Feature temporali da pickup_datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["pickup_datetime"].dt.month

    # 5. Rimuovi colonne datetime originali
    df = df.drop(columns=["pickup_datetime", "dropoff_datetime"])

    # 6. Salva output (senza index)
    df.to_csv(args.output, index=False)
    print(f"Output salvato: {args.output} ({len(df)} righe, {len(df.columns)} colonne)", file=sys.stderr)


if __name__ == "__main__":
    main()
