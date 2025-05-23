import os
import yaml
import great_expectations as gx
from pprint import pprint


def create_data_source_and_asset(context, base_directory, data_source_name, data_asset_name):
    # create a new data source, or update it if it already exists
    print("\n")
    print("Creating data source and data asset...")

    datasource = context.sources.add_or_update_pandas_filesystem(
        name=data_source_name,
        base_directory=base_directory
    )

    # add new data asset to the data source
    data_asset = datasource.add_csv_asset(
        name=data_asset_name,
        batching_regex=r"(?P<station>.*)\.csv"
    )

    # list existing data sources and data assets
    print("\nData sources:")
    pprint(context.datasources)

    print("\nAsset loaded:")
    pprint(data_asset)

    return data_asset


def define_validations(context, data_asset, expectation_suite_name, base_dir):
    batch_request = data_asset.build_batch_request()
    batch_list = context.get_batch_list(
        batch_request=batch_request
    )

    validations = [
        {
            "batch_request": batch.batch_request,
            "expectation_suite_name": expectation_suite_name
        }
        for batch in batch_list
    ]

    print("\n")
    print("Validation batches:")
    pprint([batch.batch_request.options for batch in batch_list])

    print("\n")
    print("Verifying resolved files before validation:")
    print("Working directory:", os.getcwd())
    print("Listing data dir:")
    print(os.listdir(os.path.abspath(base_dir)))
    print("\n")

    return validations


def print_results(checkpoint_result):
    print("\n")
    # extract validation result and station name for each validated station
    for _, validation_outcome in checkpoint_result["run_results"].items():
        validation_result = validation_outcome["validation_result"]
        station = validation_result["meta"]["active_batch_definition"]["batch_identifiers"].get("station", "unknown")
        success = validation_result["success"]

        print(f"Validation {'PASSED' if success else 'FAILED'} for station: {station}")

        if not success:
            print("Failed expectations:")
            for result in validation_result["results"]:
                if not result["success"]:
                    expectation_type = result["expectation_config"]["expectation_type"]
                    column = result["expectation_config"]["kwargs"].get("column", "[non-column]")
                    details = result["expectation_config"]["kwargs"]
                    print(f" - {expectation_type} on column '{column}'")
                    print(f"   failed with parameters: {details}")


def validate_temperature_data():
    
    params = yaml.safe_load(open("../params.yaml"))["validate"]
    base_dir = params["base_dir"]
    data_source_name = params["data_source_name"]
    data_asset_name = params["data_asset_name"]
    checkpoint_name = params["checkpoint_name"]
    checkpoint_run_id = params["checkpoint_run_id"]
    expectation_suite_name = params["expectation_suite_name"]

    # get context
    context = gx.get_context()

    # make base directory absolute
    base_directory = os.path.abspath(base_dir)
    print(f"Resolved base_directory: {base_directory}")

    data_asset = create_data_source_and_asset(context, base_directory, data_source_name, data_asset_name)

    # load checkpoint
    checkpoint = context.get_checkpoint(checkpoint_name)

    print(f"\nCheckpoint:\n{checkpoint}")

    validations = define_validations(context, data_asset, expectation_suite_name, base_dir)

    # run the checkpoint
    checkpoint_result = checkpoint.run(
        expectation_suite_name=expectation_suite_name,
        validations=validations,
        run_id=checkpoint_run_id
    )

    print_results(checkpoint_result)

    # build data docs
    print("\nBuilding data docs...")
    context.build_data_docs()

    # check if all validations passed
    if checkpoint_result["success"]:
        print("\nAll validations passed!")
    else:
        print("\nSome validations failed!")


if __name__ == "__main__":
    validate_temperature_data()