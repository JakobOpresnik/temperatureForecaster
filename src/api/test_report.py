import os
import glob
from fastapi import HTTPException
import yaml


def get_latest_test_reports(static_path: str) -> list[str]:
    params = yaml.safe_load(open("../../params.yaml"))
    stations = params["stations"]

    report_paths: list[str] = []

    for station_name in stations:
        reports_dir = os.path.join(static_path, "reports", station_name)
        if not os.path.exists(reports_dir):
            raise HTTPException(status_code=404, detail="Station folder not found")

        # get all HTML report files
        report_files = glob.glob(os.path.join(reports_dir, "*.html"))

        if not report_files:
            raise HTTPException(status_code=404, detail="No report files found")

        # sort by creation time descending, get latest report file
        latest_file = max(report_files, key=os.path.getctime)

        # convert to public-facing URL
        relative_path = os.path.relpath(latest_file, static_path)
        url_path = f"/static/{relative_path.replace(os.sep, '/')}"
        report_paths.append(url_path)
    
    return report_paths


def get_latest_test_report(station_name: str, static_path: str) -> str:
    reports_dir = os.path.join(static_path, "reports", station_name)
    if not os.path.exists(reports_dir):
        raise HTTPException(status_code=404, detail="Station folder not found")

    # get all HTML report files
    report_files = glob.glob(os.path.join(reports_dir, "*.html"))

    if not report_files:
        raise HTTPException(status_code=404, detail="No report files found")

    # sort by creation time descending, get latest report file
    latest_file = max(report_files, key=os.path.getctime)

    # convert to public-facing URL
    relative_path = os.path.relpath(latest_file, static_path)
    url_path = f"/static/{relative_path.replace(os.sep, '/')}"
    return url_path