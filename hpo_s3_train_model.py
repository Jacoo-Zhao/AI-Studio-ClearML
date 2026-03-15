import matplotlib.pyplot as plt
from clearml import Task, Logger, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import time
import os
import pandas as pd
import logging
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('assets', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Initialize the task
task = Task.init(
    project_name='AI_Studio_Demo',
    task_name='Pipeline step 3 train model',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'processed_dataset_id': '',
    'test_queue': 'pipeline',
    'num_epochs': 20,
    'batch_size': 16,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID from pipeline parameters
dataset_id = task.get_parameter('processed_dataset_id')  # Try without namespace first
if not dataset_id:
    dataset_id = task.get_parameter('General/processed_dataset_id')  # Try with namespace
    logger.info(f"Got dataset ID from General namespace: {dataset_id}")

logger.info(f"Received dataset ID from parameters: {dataset_id}")

if not dataset_id:
    logger.error("Processed dataset ID not found in parameters. Please ensure it's passed from the pipeline.")
    raise ValueError("Processed dataset ID not found in parameters. Please ensure it's passed from the pipeline.")

# Verify dataset exists
try:
    dataset = Dataset.get(dataset_id=dataset_id)
    logger.info(f"Successfully verified dataset: {dataset.name}")
except Exception as e:
    logger.error(f"Failed to verify dataset: {e}")
    raise

# Get the dataset files
dataset_path = dataset.get_local_copy()
logger.info(f"Dataset downloaded to: {dataset_path}")

# Load the data
X_train = pd.read_csv(os.path.join(dataset_path, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(dataset_path, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(dataset_path, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(dataset_path, 'y_test.csv'))

# Clean up temporary files
for file in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
    try:
        os.remove(os.path.join(dataset_path, file))
        logger.info(f"Cleaned up temporary directory: {file}")
    except Exception as e:
        logger.warning(f"Failed to clean up {file}: {e}")

# Convert to numpy arrays
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args['batch_size'],
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args['batch_size'],
    shuffle=False
)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=X_train.shape[1], num_classes=len(set(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=args['learning_rate'],
    weight_decay=args['weight_decay']
)

# Training loop
for epoch in tqdm(range(args['num_epochs']), desc="Training Epochs"):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    # Report training loss
    task.get_logger().report_scalar(
        title='train',
        series='epoch_loss',
        value=avg_loss,
        iteration=epoch
    )
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    accuracy = 100 * correct / total
    # Report validation accuracy
    task.get_logger().report_scalar(
        title='validation',
        series='accuracy',
        value=accuracy,
        iteration=epoch
    )
    # Also report as a single value for HPO
    task.get_logger().report_scalar(
        title='validation',
        series='accuracy',
        value=accuracy,
        iteration=0
    )

# Save the model
model_path = os.path.join(os.getcwd(), 'model.pth')
torch.save(model.state_dict(), model_path)
task.upload_artifact('model', model_path)

print('Training completed successfully')

# Plotting confusion matrix
species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
y_test_names = [species_mapping[label] for label in all_targets]
predicted_names = [species_mapping[label] for label in all_predictions]

cm = confusion_matrix(y_test_names, predicted_names, labels=list(species_mapping.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(species_mapping.values()))
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.savefig('figs/confusion_matrix.png')

print('Confusion matrix plotted and saved as confusion_matrix.png')