
# task.connect(args)
#
# # only create the task, we will actually execute it later
# # task.execute_remotely() # After passing local testing, you should uncomment this command to initial task to ClearML
#
# print('Retrieving Iris dataset')
# dataset_task = Task.get_task(task_id=args['dataset_task_id'])
# X_train = dataset_task.artifacts['X_train'].get()
# X_test = dataset_task.artifacts['X_test'].get()
# y_train = dataset_task.artifacts['y_train'].get()
# y_test = dataset_task.artifacts['y_test'].get()
# print('Iris dataset loaded')
#
#
# # Define a simple neural network
# class SimpleNN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 50)
#         self.fc2 = nn.Linear(50, num_classes)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # Convert data to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)
#
# # Create DataLoader
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
# # Hyperparameters
# # Initialize the model, loss function, and optimizer
# model = SimpleNN(input_size=X_train.shape[1], num_classes=len(set(y_train)))
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(
#     model.parameters(),
#     lr=args['learning_rate'],
#     weight_decay=args['weight_decay']
# )
#
# for epoch in tqdm(range(args['num_epochs']), desc='Training Epochs'):
#     epoch_loss = 0.0
#
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # 累积 loss
#         epoch_loss += loss.item()
#
#     avg_loss = epoch_loss / len(train_loader)
#     logger.report_scalar(title='train', series='epoch_loss', value=avg_loss, iteration=epoch)
#
# # Save model
# model_path = 'assets/model.pkl'
# torch.save(model.state_dict(), model_path)
# task.upload_artifact(name='model', artifact_object=model_path)
# print('Model saved and uploaded as artifact')
#
# # Load model for evaluation
# model.load_state_dict(torch.load(model_path))
# model.eval()
# with torch.no_grad():
#     outputs = model(X_test_tensor)
#     _, predicted = torch.max(outputs, 1)
#     accuracy = (predicted == y_test_tensor).float().mean().item()
#     logger.report_scalar("validation_accuracy", "score", value=accuracy, iteration=0)
#
# print(f'Model trained & stored with accuracy: {accuracy:.4f}')
#
#
# # Plotting confusion matrix
# species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
# y_test_names = [species_mapping[label.item()] for label in y_test]
# predicted_names = [species_mapping[label.item()] for label in predicted]
#
# cm = confusion_matrix(y_test_names, predicted_names, labels=list(species_mapping.values()))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(species_mapping.values()))
# disp.plot(cmap=plt.cm.Blues)
#
# plt.title('Confusion Matrix')
# plt.savefig('figs/confusion_matrix.png')
#
# print('Confusion matrix plotted and saved as confusion_matrix.png')

import os
os.environ.pop('MPLBACKEND', None)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task, Logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(project_name="AI_Studio_Basic_Demo", task_name="Pipeline step 3 train model")
logger = Logger.current_logger()

# Arguments
args = {
    'dataset_task_id': '', # replace the value only when you need debug locally
}
task.connect(args)

# only create the task, we will actually execute it later
task.execute_remotely() # After passing local testing, you should uncomment this command to initial task to ClearML

print('Retrieving Iris dataset')
dataset_task = Task.get_task(task_id=args['dataset_task_id'])
X_train = dataset_task.artifacts['X_train'].get()
X_test = dataset_task.artifacts['X_test'].get()
y_train = dataset_task.artifacts['y_train'].get()
y_test = dataset_task.artifacts['y_test'].get()
print('Iris dataset loaded')

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

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleNN(input_size=X_train.shape[1], num_classes=len(set(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.report_scalar(title='train', series='loss', value=loss.item(), iteration=epoch)

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).float().mean().item()

print(f'Model trained & stored with accuracy: {accuracy:.4f}')

# Plotting (same as before)
x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plt.figure(1, figsize=(4, 3))


plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.title('Iris Types')
plt.savefig('iris_plot.png')

print('Done🔥')