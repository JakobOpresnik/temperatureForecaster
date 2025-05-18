import os
import sys
import great_expectations as gx
from pprint import pprint

context = gx.get_context()

# make base directory absolute
base_directory = os.path.abspath("../data/preprocessed/temp/")
print(f"Resolved base_directory: {base_directory}")

# create a new data source, or update it if it already exists
datasource_name = "temperature"
datasource = context.sources.add_or_update_pandas_filesystem(
    name=datasource_name,
    base_directory=base_directory
)

# add new data asset to the data source
data_asset_name = "temperature_data"
data_asset = datasource.add_csv_asset(
    name=data_asset_name,
    batching_regex=r"(?P<station>.*)\.csv"
)

# list existing data sources
print(context.datasources)

# load the data asset
asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

print(f"Asset loaded:\n{asset}")

# load checkpoint
checkpoint_name = "temperature_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

print(f"Checkpoint:\n{checkpoint}")

batch_request = asset.build_batch_request()
batch_list = context.get_batch_list(
    batch_request=batch_request
)

validations = [
    {
        "batch_request": batch.batch_request,
        "expectation_suite_name": "temperature_suite"
    }
    for batch in batch_list
]

print("\nValidation batches:")
pprint([batch.batch_request.options for batch in batch_list])
print("\n")

print("Verifying resolved files before validation:")
print("Working directory:", os.getcwd())
print("Listing data dir:")
print(os.listdir(os.path.abspath("../data/preprocessed/temp/")))

# run the checkpoint
run_id = "temperature_run"
checkpoint_result = checkpoint.run(
    expectation_suite_name="temperature_suite",
    validations=validations,
    run_id=run_id
)

print("\n")

# extract validation result and station name for each validated station
for batch_id, validation_outcome in checkpoint_result["run_results"].items():
    validation_result = validation_outcome["validation_result"]
    station = validation_result["meta"]["active_batch_definition"]["batch_identifiers"].get("station", "unknown")
    success = validation_result["success"]
    print(f"Validation {'passed' if success else 'failed'} for station: {station}")

# build data docs
print("\nBuilding data docs...")
context.build_data_docs()

# check if all validations passed
if checkpoint_result["success"]:
    print("\nAll validations passed!")
    sys.exit(0)
else:
    print("\nSome validations failed!")
    sys.exit(1)