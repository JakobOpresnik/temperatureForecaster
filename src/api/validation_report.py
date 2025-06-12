import glob
import os
from fastapi import HTTPException
from pathlib import Path
from datetime import datetime
import yaml


def parse_folder_timestamp(folder_name: str) -> datetime:
    try:
        return datetime.strptime(folder_name, "%Y%m%dT%H%M%S.%fZ")
    except ValueError:
        return datetime.min


def get_latest_validation_reports(static_path: str) -> list[str]:
    params = yaml.safe_load(open("../../params.yaml"))
    stations = params["stations"]

    validation_root = Path(static_path) / "validation/gx/uncommitted/data_docs/local_site/validations/temperature_suite/__none__"
    if not validation_root.exists():
        raise HTTPException(status_code=500, detail=f"Validation root path not found: {validation_root}")

    subfolders: list[Path] = [f for f in validation_root.iterdir() if f.is_dir()]
    subfolders.sort(key=lambda f: parse_folder_timestamp(f.name), reverse=True)

    report_paths: list[str] = []
    found_stations = set()

    for folder in subfolders:
        for station in stations:
            if station in found_stations:
                continue
            html_file = folder / f"temperature-temperature_data-station_{station}.html"
            if html_file.exists():
                relative_path = os.path.relpath(html_file, static_path)
                url_path = f"/static/{relative_path.replace(os.sep, '/')}"
                report_paths.append(url_path)
                found_stations.add(station)
        if len(found_stations) == len(stations):
            break

    return report_paths