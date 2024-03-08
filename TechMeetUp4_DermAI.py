#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries#


# In[33]:


from torchvision import transforms, datasets, models
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.optim as optim
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


# #Transforms and DataLoader/Augmentation#

# In[3]:


zip_file_path = '/content/medicalData.zip'
target_dir = '/content/dataset'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

print("Extraction done!")


# In[7]:


# Augmentation + transformations for the training data
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  #
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[8]:


base_dir = '/content/dataset/data'

# Training data
train_dir = os.path.join(base_dir, 'train')
train_dataset = ImageFolder(train_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Testing data
test_dir = os.path.join(base_dir, 'test')
test_dataset = ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Training and testing datasets are loaded.")


# In[32]:


# Function to unnormalize and display an image
def imshow(img, ax=None):
    """Imshow for Tensor."""
    img = img.cpu().numpy().transpose((1, 2, 0))  # transpose back to height*width*channels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # unnormalize
    img = np.clip(img, 0, 1)  # clip any values outside the range [0,1]
    if ax is None:
        ax = plt.gca()
    ax.imshow(img)
    ax.axis('off')

class_names = train_dataset.classes
class_images = {class_name: None for class_name in class_names}

# Load images, making sure to get one per class
for images, labels in train_loader:
    for image, label in zip(images, labels):
        class_name = class_names[label]
        if class_images[class_name] is None:
            class_images[class_name] = image
        # Check if we have one image for each class
        if all(class_images[class_name] is not None for class_name in class_names):
            break
    if all(class_images[class_name] is not None for class_name in class_names):
        break

# Make sure we're only plotting if we have all the images
if not all(class_images[class_name] is not None for class_name in class_names):
    print("Could not find an image for each class.")
else:
    # Plot the images
    fig = plt.figure(figsize=(15, 5))
    for i, (class_name, image) in enumerate(class_images.items()):
        ax = fig.add_subplot(1, len(class_names), i + 1, xticks=[], yticks=[])
        imshow(image, ax=ax)
        ax.set_title(class_name)
    plt.show()


# #Initializing & modify Model#

# In[9]:


model = models.vgg16(pretrained=True)


# In[12]:


# Modify the classifier to fit the 3-class problem
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 3)])
model.classifier = nn.Sequential(*features)

for param in model.features.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

epochs = 20


# #Training#

# In[17]:


train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Predictions
        _, preds = torch.max(outputs, 1)

        # Calculate loss
        loss = criterion(outputs, labels)

         # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / total_predictions

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# #Testing#

# In[18]:


model.eval()

running_loss = 0.0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)


average_loss = running_loss / len(test_loader)
accuracy = correct_predictions / total_predictions

print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')


# In[37]:


running_loss = 0.0
correct_predictions = 0
total_predictions = 0

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Predictions
        _, predicted = torch.max(outputs.data, 1)

        # Correct predictions
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        # Append batch predictions and labels to the list for classification report
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


average_loss = running_loss / len(test_loader)
accuracy = correct_predictions / total_predictions


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)


# In[38]:


print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')


# In[39]:


report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
print(report)


# #Plot Training Loss & Accuracy#

# In[21]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()

plt.show()

