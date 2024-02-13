# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

"""
import mlflow
import os
import requests
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import shutil


# Initialize MLflow
mlflow.set_tracking_uri('https://0c4a-34-91-178-206.ngrok-free.app/')
exp_id = mlflow.get_experiment_by_name("mlops_exp").experiment_id
mlflow.set_experiment(experiment_id=exp_id)
mlflow.start_run()


# function to download images from urls
def download_images(urls, location: str, suffix):
  """function to download images from urls"""
  for i, url in enumerate(urls):
    try:
      response = requests.get(url, timeout=15)
    except:
      continue

    path = location + suffix + "_" + str(i) + ".jpg"
    with open(path, "wb") as f:
      f.write(response.content)

def load_data(file_name):
  """Load Data List"""
  with open("./data/{}.json".format(file_name), "r") as fp:
      search_result_links_re_read = json.load(fp)
  return search_result_links_re_read


shutil.rmtree('photos', True)
os.makedirs(os.path.dirname("photos/dogs/"), exist_ok=True)
os.makedirs(os.path.dirname("photos/cats/"), exist_ok=True)
os.makedirs(os.path.dirname("photos/cars/"), exist_ok=True)


dgimages = load_data('pet_dog')
download_images(dgimages, "photos/dogs/", "dog")

ctimages = load_data('pet_cat')
download_images(ctimages, "photos/cats/", "cat")
# classes_num = 2

carimages = load_data('super_car')
download_images(carimages, "photos/cars/", "car")
classes_num = 3

# filtering and deleting very small files
for path, subdirs, files in os.walk("photos"):
    for name in files:
      file = os.path.join(path, name)
      if os.path.getsize(file) < 24 * 1024:
        os.remove(file)


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageFolder("./photos", transform=data_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)


model = torchvision.models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, classes_num)  # 2 classes: cat and dog

lr = 0.001
momentum = 0.9
num_epochs = 8

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr, momentum=momentum)
device = torch.device("cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")


    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

# Save the model as an MLflow artifact
mlflow.log_params({"lr": lr})
mlflow.log_params({"momentum": momentum})
mlflow.log_params({"num_epochs": num_epochs})
mlflow.log_metric("val_accuracy", accuracy, step=epoch)
mlflow.log_metric("train_loss", avg_loss, step=epoch)
mlflow.pytorch.log_model(model, "cat_dog_resnet_model")
mlflow.end_run()
