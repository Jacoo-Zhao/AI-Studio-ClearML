"s1_dataset_artifact.py"

from clearml import Task, StorageManager
import os

# create an dataset experiment
task = Task.init(project_name="AI_Studio_Basic_Demo", task_name="Pipeline step 1 dataset artifact")

# only create the task, we will actually execute it later
task.execute_remotely()


local_iris_pkl = StorageManager.get_local_copy(remote_url='https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl')

# Add and upload the dataset file
task.upload_artifact('dataset', artifact_object=local_iris_pkl)

print('Uploading artifacts in the background')
print('Step 1 Completed🔥')