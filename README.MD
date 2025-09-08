install dependencies
set up settings
run download_sron_ftp.py (# run as python src\config\ingest\download_sron_ftp.py --years 2024)
download Kayrros Methane data
run merge.py
run qa_provider_checks.py
run_dedup.py
run merge.py
run cluster.py
