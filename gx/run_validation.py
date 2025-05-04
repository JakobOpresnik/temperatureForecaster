import sys
import great_expectations as gx

context = gx.get_context()

datasource_name = "temperature"
data_asset_name = "temperature_data"

# load the data asset
asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

# load checkpoint
checkpoint_name = "temperature_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

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