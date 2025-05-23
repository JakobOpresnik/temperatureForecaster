import os
import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset
import yaml
from datetime import datetime


def clean_data(reference_data, current_data):
    # no need to compare dates
    del reference_data["Date"]
    del current_data["Date"]

    # drop completely empty columns if they occur
    reference_data.dropna(axis=1, how='all', inplace=True)
    current_data.dropna(axis=1, how='all', inplace=True)


def save_test_report(reports_file_path_template, station, result):
    # create folder to save report if it doesn't exist yet
    reports_path = reports_file_path_template.format(station=station)
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    # save report to HTML file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result.save_html(f"{reports_path}/data_testing_report_{station}_{timestamp}.html")


def check_test_results(result, station, reference_path, current_path):
    # check if report contains any tests and if all tests passed
    all_tests_passed = True
    result_dict = result.dict()

    # print test results
    if "tests" in result_dict:
        for test in result_dict["tests"]:
            if "status" in test and test["status"] != "SUCCESS":
                all_tests_passed = False
                if "name" in test:
                    print(f"FAILED TEST: {test["name"].split(":")[-1]}")
                if "description" in test:
                    print(test["description"].split(":")[-1])
                break
                
    if not all_tests_passed:
        print(f"Data tests failed for station: {station}\n")
    else:
        print(f"Data tests passed for station: {station}")
        # replace reference data with current data
        os.remove(reference_path)
        current = pd.read_csv(current_path)
        current.to_csv(reference_path, index=False)


def test_temperature_data():
    print("Current working directory:", os.getcwd())

    params_preprocessed = yaml.safe_load(open("params.yaml"))["preprocess"]
    params_test = yaml.safe_load(open("params.yaml"))["test"]
    stations = yaml.safe_load(open("params.yaml"))["stations"]

    output_file_path_template = params_preprocessed["output_file_path_template"]
    reference_file_path_template = params_test["reference_file_path_template"]
    reports_file_path_template = params_test["reports_file_path_template"]

    for station in stations:
        current_path = output_file_path_template.format(station=station)
        current_data = pd.read_csv(current_path)
        reference_path = reference_file_path_template.format(station=station)

        path_exists = os.path.exists(reference_path)
        if not path_exists:
            print(f"Reference file not found. Copying from current data to {reference_path}.")
            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            current_data.to_csv(reference_path, index=False)

        reference_data = pd.read_csv(reference_path)

        clean_data(reference_data, current_data)

        # define drift detection parameters
        drift_preset = DataDriftPreset(
            drift_share=0.7,     # fraction of columns that must drift to trigger overall drift (lower means stricter)
            num_threshold=0.05,  # fraction of rows (numerical values) inside a column that must drift to trigger column drift (p-value --> lower means stricter)
        )

        # check if reference and current data have the same columns
        report = Report([
                DataSummaryPreset(),
                drift_preset
            ],
            include_tests=True
        )

        # run report on reference and current data
        result = report.run(reference_data=reference_data, current_data=current_data)

        save_test_report(reports_file_path_template, station, result)
        check_test_results(result, station, reference_path, current_path)


if __name__ == "__main__":
    test_temperature_data()