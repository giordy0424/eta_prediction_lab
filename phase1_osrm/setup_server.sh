#!/usr/bin/env bash
set -euo pipefail

OSM_EXTRACT="${1:-manhattan.osm.pbf}"
OSRM_DIR="${2:-./osrm_data}"

mkdir -p "$OSRM_DIR"

if [ ! -f "${OSRM_DIR}/${OSM_EXTRACT}" ]; then
    echo "ERROR: OSM extract not found at ${OSRM_DIR}/${OSM_EXTRACT}"
    echo "Download from https://extract.bbbike.org/ and place in ${OSRM_DIR}/"
    exit 1
fi

echo "=== Phase 1: OSRM Server Setup ==="

echo "[1/3] Extracting road graph..."
docker run --rm -v "${PWD}/${OSRM_DIR}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua "/data/${OSM_EXTRACT}"

echo "[2/3] Contracting graph (Contraction Hierarchies)..."
docker run --rm -t -v "${PWD}/${OSRM_DIR}:/data" osrm/osrm-backend osrm-contract "/data/${OSM_EXTRACT%.osm.pbf}.osrm"

echo "[3/3] Starting routing server on :5000..."
docker run --name osrm_server -d -t -i \
    -p 5000:5000 \
    -v "${PWD}/${OSRM_DIR}:/data" \
    osrm/osrm-backend osrm-routed --algorithm ch "/data/${OSM_EXTRACT%.osm.pbf}.osrm"

echo "Server started at http://localhost:5000"
echo "Test: curl 'http://localhost:5000/route/v1/driving/-73.985754,40.748128;-74.012665,40.713240?steps=true'"
