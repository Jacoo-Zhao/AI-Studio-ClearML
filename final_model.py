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
import json
import numpy as np
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('assets', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Initialize the task
task = Task.init(
    project_name='AI_Studio_Demo',
    task_name='Final Model Training',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'processed_dataset_id': '99e286d358754697a37ad75c279a6f0a',  # Will be set from pipeline
    'hpo_task_id': None,  # Will be set from pipeline
    'test_queue': 'pipeline',  # Queue for test tasks
    'num_epochs': 50,  # Will be overridden by best HPO parameters
    'batch_size': 32,  # Will be overridden by best HPO parameters
    'learning_rate': 1e-3,  # Will be overridden by best HPO parameters
    'weight_decay': 1e-5  # Will be overridden by best HPO parameters
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID from pipeline parameters
dataset_id = task.get_parameter('General/processed_dataset_id')  # Get from General namespace
if not dataset_id:
    # Try getting from args as fallback
    dataset_id = args.get('processed_dataset_id')
    print(f"No dataset_id now get dataset ID from args: {dataset_id}")

if not dataset_id:
    # Use fixed dataset ID as last resort
    dataset_id = "99e286d358754697a37ad75c279a6f0a"
    print(f"Using fixed dataset ID: {dataset_id}")

logger.info(f"Received dataset ID from parameters: {dataset_id}")

if not dataset_id:
    logger.error("Processed dataset ID not found in parameters. Please ensure it's passed from the pipeline.")
    raise ValueError("Processed dataset ID not found in parameters. Please ensure it's passed from the pipeline.")

# Get the HPO task ID
hpo_task_id = args.get('hpo_task_id')
if not hpo_task_id:
    logger.error("HPO task ID not found in parameters")
    raise ValueError("HPO task ID not found in parameters")

# Get the HPO task
hpo_task = Task.get_task(task_id=hpo_task_id)
logger.info(f"Retrieved HPO task: {hpo_task.name}")

# Get best parameters
try:
    # First try to get from task parameters
    best_params = hpo_task.get_parameter('best_parameters')
    best_accuracy = hpo_task.get_parameter('best_accuracy')
    
    if best_params is None:
        # If not in parameters, try to get from artifact
        logger.info("Best parameters not found in task parameters, trying artifact...")
        if 'best_parameters' not in hpo_task.artifacts:
            logger.error("No best_parameters artifact found in HPO task")
            raise ValueError("No best_parameters artifact found in HPO task")
            
        artifact_path = hpo_task.artifacts['best_parameters'].get_local_copy()
        if artifact_path is None:
            logger.error("Failed to get local copy of best_parameters artifact")
            raise ValueError("Failed to get local copy of best_parameters artifact")
            
        logger.info(f"Downloaded best parameters from: {artifact_path}")
        
        with open(artifact_path, 'r') as f:
            best_results = json.load(f)
        
        best_params = best_results['parameters']
        best_accuracy = best_results.get('accuracy')
    
    # Update training parameters with best values
    args['num_epochs'] = best_params.get('num_epochs', args['num_epochs'])
    args['batch_size'] = best_params.get('batch_size', args['batch_size'])
    args['learning_rate'] = best_params.get('learning_rate', args['learning_rate'])
    args['weight_decay'] = best_params.get('weight_decay', args['weight_decay'])
    
    logger.info(f"Using best parameters from HPO: {best_params}")
    logger.info(f"Best validation accuracy from HPO: {best_accuracy}")
except Exception as e:
    logger.error(f"Failed to get best parameters from HPO task: {e}")
    raise

# Verify dataset exists
try:
    dataset = Dataset.get(dataset_id=dataset_id)
    logger.info(f"Successfully verified dataset: {dataset.name}")
except Exception as e:
    logger.error(f"Failed to verify dataset: {e}")
    raise

# Load the data
try:
    # Get the dataset path
    dataset_path = dataset.get_local_copy()
    logger.info(f"Dataset downloaded to: {dataset_path}")
    
    # Load training and testing data from separate files
    X_train = pd.read_csv(os.path.join(dataset_path, 'X_train.csv')).values
    X_test = pd.read_csv(os.path.join(dataset_path, 'X_test.csv')).values
    y_train = pd.read_csv(os.path.join(dataset_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(dataset_path, 'y_test.csv')).values.ravel()

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    
    logger.info(f"Data loaded successfully. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, len(set(y_train.numpy())))  # In case labels change
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN(X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

# Training loop
logger.info("Starting training...")
for epoch in range(args['num_epochs']):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Report training loss
    avg_loss = running_loss / len(train_loader)
    task.get_logger().report_scalar('training', 'loss', value=avg_loss, iteration=epoch)
    logger.info(f'Epoch {epoch+1}/{args["num_epochs"]}, Loss: {avg_loss:.4f}')
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = 100 * correct / total
    task.get_logger().report_scalar('validation', 'accuracy', value=accuracy, iteration=epoch)
    logger.info(f'Validation Accuracy: {accuracy:.2f}%')

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
task.get_logger().report_matplotlib_figure('Confusion Matrix', 'confusion_matrix', plt.gcf(), epoch)

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')
task.upload_artifact('model', 'final_model.pth')
logger.info("Model saved and uploaded as artifact")

print('Training completed successfully!') 