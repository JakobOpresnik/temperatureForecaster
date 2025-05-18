import sys
import great_expectations as gx
from pprint import pprint


context = gx.get_context()

datasource_name = "temperature"
data_asset_name = "temperature_data"

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

print("\nValidation batches:")
pprint([batch.batch_request.options for batch in batch_list])
print("\n")

# run the checkpoint
run_id = "temperature_run"
checkpoint_result = checkpoint.run(
    run_id=run_id
)

# build data docs
context.build_data_docs()

# check if checkpoint passed
if checkpoint_result["success"]:
    print("Validation passed!")
    sys.exit(0)
else:
    print("Validation failed!")
    sys.exit(1)



""" import sys
import great_expectations as gx
from pprint import pprint

from pathlib import Path
print("Resolved path:", Path("../data/preprocessed/temp/").resolve())

context = gx.get_context()

datasource_name = "temperature"
data_asset_name = "temperature_data"
expectation_suite_name = "temperature_suite"

# Load the data asset
asset = context.get_datasource(datasource_name).get_asset(data_asset_name)
print("Asset loaded:", asset)

# Discover all batches (i.e., all stations)
batch_request = asset.build_batch_request()
batch_list = context.get_batch_list(
    batch_request=batch_request
)

print(f"Discovered {len(batch_list)} batches")
print(batch_list)

pprint([batch.batch_request.options for batch in batch_list])

# Build validations for each batch (each station)
validations = []
for batch in batch_list:
    station = batch.batch_request.options["station"]
    print(f"Adding validation for station: {station}")
    validations.append({
        "batch_request": asset.build_batch_request(options={"station": station}).to_json_dict(),
        "expectation_suite_name": expectation_suite_name
    })

# Load checkpoint
checkpoint = context.get_checkpoint("temperature_checkpoint")
print("Checkpoint loaded.")

# Run checkpoint with validations for all stations
checkpoint_result = checkpoint.run(
    validations=validations,
    run_id="temperature_run"
)

# Build data docs
context.build_data_docs()

# Check if checkpoint passed for all
if checkpoint_result["success"]:
    print("✅ Validation passed for all stations!")
    sys.exit(0)
else:
    print("❌ Validation failed for at least one station.")
    sys.exit(1)
 """