from clearml import Task, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the task
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 2 process dataset")

# Connect parameters
args = {
    'dataset_task_id': '',  # Will be set from pipeline
    'test_size': 0.25,
    'random_state': 42
}
task.connect(args)

# Execute the task remotely
task.execute_remotely()

# Get the dataset task ID from pipeline parameters
dataset_task_id = task.get_parameter('General/dataset_task_id')
logger.info(f"Using dataset task ID: {dataset_task_id}")

# Load the raw dataset from ClearML
dataset_task = Task.get_task(task_id=dataset_task_id)
raw_dataset = Dataset.get(dataset_id=dataset_task.get_parameter('General/dataset_id'))
logger.info(f"Loaded raw dataset: {raw_dataset.name}")

# Get the raw data
dataset_path = raw_dataset.get_mutable_local_copy("iris_dataset.csv")
raw_data = pd.read_csv(os.path.join(dataset_path, "iris_dataset.csv"))
logger.info("Successfully loaded raw data")

# Process the data
X = raw_data.drop('target', axis=1)
y = raw_data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state']
)

# Create a new dataset in ClearML
dataset = Dataset.create(
    dataset_name="Iris Processed Dataset",
    dataset_project="AI_Studio_Demo"
)

# Save processed data to temporary files
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Add the processed data to the dataset
for file in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    dataset.add_files(file)
    logger.info(f"Added {file} to dataset")

# Upload the dataset
dataset.upload()
logger.info("Uploaded dataset files")

# Finalize the dataset
dataset.finalize()
logger.info(f"Dataset created with ID: {dataset.id}")

# Store the dataset ID as a task parameter for other steps to use
task.set_parameter("General/processed_dataset_id", str(dataset.id))  # Convert to string to ensure proper passing
logger.info(f"Stored processed dataset ID: {dataset.id}")

# Verify the parameter was set correctly
stored_id = task.get_parameter("General/processed_dataset_id")
logger.info(f"Verified stored dataset ID: {stored_id}")

# Also store as a task artifact
task.upload_artifact(name='processed_dataset_id', artifact_object=str(dataset.id))
logger.info(f"Stored dataset ID as task artifact: {dataset.id}")

# Clean up temporary files
for file in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
            logger.info(f"Cleaned up temporary directory: {file}")
        else:
            os.remove(file)
            logger.info(f"Cleaned up temporary file: {file}")

print("Dataset processing completed and uploaded to ClearML") 