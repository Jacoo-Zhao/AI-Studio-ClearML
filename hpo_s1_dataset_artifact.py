"s1_dataset_artifact.py"

from clearml import Task, Dataset
import pandas as pd
from sklearn.datasets import load_iris
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the task
task = Task.init(project_name="AI_Studio_Demo", task_name="Pipeline step 1 dataset artifact")

# only create the task, we will actually execute it later
task.execute_remotely()

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Create a new dataset in ClearML
dataset = Dataset.create(
    dataset_name="Iris Raw Dataset",
    dataset_project="AI_Studio_Demo"
)

# Save the data to a temporary file
temp_file = "iris_dataset.csv"
data.to_csv(temp_file, index=False)
logger.info(f"Saved data to temporary file: {temp_file}")

# Add the data to the dataset
dataset.add_files(temp_file)
logger.info("Added data file to dataset")

# Upload the dataset
dataset.upload()
logger.info("Uploaded dataset files")

# Finalize the dataset
dataset.finalize()
logger.info(f"Dataset created with ID: {dataset.id}")

# Store the dataset ID as a task parameter for other steps to use
task.set_parameter("General/dataset_id", dataset.id)
logger.info(f"Stored dataset ID: {dataset.id}")

# Clean up temporary file
os.remove(temp_file)
logger.info("Cleaned up temporary file")

print("Dataset created and uploaded to ClearML")