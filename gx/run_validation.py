import os
import sys
import great_expectations as gx
from pprint import pprint

context = gx.get_context()

# remove existing data source, if needed
context.sources.delete_pandas_filesystem("temperature")

# make base directory absolute
base_directory = os.path.abspath("../data/preprocessed/temp/")
print(f"Resolved base_directory: {base_directory}")

# create a new data source
datasource_name = "temperature"
datasource = context.sources.add_pandas_filesystem(
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


# context = gx.get_context()

datasource_name = "temperature"
data_asset_name = "temperature_data"

# load the data asset
asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

print(f"Asset loaded:\n{asset}")

# load checkpoint
checkpoint_name = "temperature_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)


# batch_request = asset.build_batch_request()
batch_request = asset.build_batch_request({"station": "BOVEC"})
batch_list = context.get_batch_list(
    batch_request=batch_request
)

print("\nValidation batches:")
pprint([batch.batch_request.options for batch in batch_list])
print("\n")


import os
base_dir = "../data/preprocessed/temp/"
print(f"Files in base directory '{base_dir}':")
for filename in os.listdir(base_dir):
    print(filename)


print("🧪 Verifying resolved files before validation:")
import os
print("Working directory:", os.getcwd())
print("Listing data dir:")
print(os.listdir(os.path.abspath("../data/preprocessed/temp/")))


print(f"Checkpoint:\n{checkpoint}")


stations = [station.split(".")[0] for station in os.listdir(os.path.abspath("../data/preprocessed/temp/"))]
print("stations: ", stations)

""" for station in stations:
    checkpoint_result = checkpoint.run(
        # run_id=f"{station}_run",
        validations=[
            {
                "expectation_suite_name": "temperature_suite",
                "batch_request": {
                    "datasource_name": datasource_name,
                    "data_asset_name": data_asset_name,
                    "options": {
                        "station": station
                    }
                }
            }
        ]
    ) """

checkpoint_result = checkpoint.run(
    validations=[
        {
            "expectation_suite_name": "temperature_suite",
            "batch_request": batch_request,
        }
    ]
)


# run the checkpoint
""" run_id = "temperature_run"
checkpoint_result = checkpoint.run(
    run_id=run_id
) """

# build data docs
context.build_data_docs()

# check if checkpoint passed
if checkpoint_result["success"]:
    print("Validation passed!")
    sys.exit(0)
else:
    print("Validation failed!")
    sys.exit(1)