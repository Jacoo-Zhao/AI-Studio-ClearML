"s2_data_preprocessing.py"

import pickle
from clearml import Task, StorageManager
from sklearn.model_selection import train_test_split

task = Task.init(project_name="AI_Studio_Basic_Demo", task_name="Pipeline step 2 process dataset")

args = {
    'dataset_task_id': '', 
    'random_state': 42,
    'test_size': 0.2,
}

task.connect(args)
print('Arguments: {}'.format(args))

task.execute_remotely()

# get dataset from task's artefact
if args['dataset_task_id']:
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_upload_task.artifacts.keys())))
    iris_pickle = dataset_upload_task.artifacts['dataset'].get_local_copy()
else:
    raise ValueError("Missing dataset link")

# open the local copy
iris = pickle.load(open(iris_pickle, 'rb'))

# Split data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state'])

# Upload processed data
print('Uploading process dataset')
task.upload_artifact('X_train', X_train)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_train', y_train)
task.upload_artifact('y_test', y_test)

print('Task 2 completed.🔥')