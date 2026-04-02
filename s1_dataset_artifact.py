"s1_dataset_artifact.py"

from clearml import Task, StorageManager
import os

# create an dataset experiment
task = Task.init(project_name="AI_Studio_Basic_Demo", task_name="Pipeline step 1 dataset artefact")

task.execute_remotely()

local_iris_pkl = StorageManager.get_local_copy(remote_url='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl')

# Upload the dataset file
task.upload_artifact('dataset', artifact_object=local_iris_pkl)

print('Step 1 completed.🔥')