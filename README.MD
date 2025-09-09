# Methane Super Emitter Data Pipeline

This repo has scripts and workflows to extract, transform and load methane super-emitter detections via satellite from multiple providers (SRON, Kayrros, Carbon Mapper). Outputs include cleaned datasets, cluster analysis, PNG outputs and a webmap built on Next.js + Maplibre hosted on Vercel.

The current configuration uses a manual download form Kayrros though this could be adapted to use an API call instead.
It downloads the Carbon Mapper output from their API (key required) and accesses the SRON data via their FTP site.

---

## Quick Start

Follow these steps to set up your environment, install dependencies, and run the full pipeline. Please note that these commands are built on
powershell and the path configuration may differ on Mac or Linux.

### 1. Create and activate a Python virtual environment

```powershell
# From repo root
py -3.11 -m venv .venv
. .venv\Scripts\Activate.ps1
```

### 2. Install dependencies

All required Python packages are listed in `requirements.txt`.

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> If you add new packages during development, update the requirements.txt with:
>
> ```powershell
> pip freeze > requirements.txt
> ```

### 3. Configure secrets

Create a `.env` file in the repository root if you plan to use the Carbon Mapper API.

### 4. Ingest SRON and Kayrros data

```powershell
# Download SRON weekly CSVs via FTP
python src\ingest\download_sron_ftp.py --years 2023 2024
```

# Place Kayrros CSVs manually under data/raw/kayrros/

### 5. Clean and unify

```powershell
python src\clean_and_unify.py `
  --sron "data\raw\sron\*.csv" `
  --kayrros "data\raw\kayrros\*.csv" `
  --clip-au
```

→ `data/interim/plumes_unified.parquet`

### 6. Add Carbon Mapper

```powershell
python src\ingest\carbon_mapper_api.py --bbox 112 -44 154 -9 --start 2019-01-01 --end 2025-12-31 --verbose
```

### 7. Merge with unified

```powershell
python src\clean_and_merge_carbonmapper.py --unified-in "data\interim\plumes_unified.parquet" --clip-au
```

→ `data/interim/unified_plus_cm.parquet`

### 8. Deduplicate across providers

```powershell
python src\dedup.py `
  --in-parquet "data\interim\unified_plus_cm.parquet" `
  --out-parquet "data\processed\plumes_dedup.parquet" `
  --out-gpkg "data\processed\plumes_dedup.gpkg" `
  --pairs-csv "data\processed\crossdup_pairs.csv"
```

### 9. Run DBSCAN clustering

```powershell
python src\cluster.py `
  --in-gpkg "data\processed\plumes_dedup.gpkg" `
  --layer "plumes_dedup" `
  --engine haversine `
  --eps-m 30000 `
  --min-samples 5 `
  --make-png
```

Outputs:

- `data/processed/au_plumes_clustered.parquet`
- `data/processed/clusters_au.gpkg`
- `outputs/maps/clusters_au.png`

### 10. Export GeoJSON for the webmap

```powershell
python src\export_geojson.py `
  --points-in "data\processed\au_plumes_clustered.parquet" `
  --provider-ref "data\interim\unified_plus_cm.parquet" `
  --cm-in "data\interim\cm_plumes_clean.geojson" `
  --clip-au
```

Writes to `webmap-app/public/data/au_plumes.geojson` and `webmap-app/public/data/clusters_au.geojson`.

### 11. Run the webmap locally

```powershell
cd webmap-app
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to view the interactive map.
