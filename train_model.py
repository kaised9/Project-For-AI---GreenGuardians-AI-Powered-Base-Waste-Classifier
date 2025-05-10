import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from wasteClassifier.models import WasteClassifier

# Step 1: Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize image values to [-1, 1]
])

# Step 2: Load dataset and split into training and validation sets
dataset = datasets.ImageFolder('data/train', transform=transform)
print(dataset.class_to_idx)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 3: Set up model, loss function, optimizer
model = WasteClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Define evaluation function
def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    return avg_loss, accuracy

# Step 5: Train the model
print("🔄 Training started...\n")
num_epochs = 5
for epoch in range(num_epochs):  
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluate the model on the validation set
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
    print(f"✅ Epoch {epoch+1} - Loss: {running_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")

# After training is complete, save the model:
torch.save(model.state_dict(), 'waste_model.pth')
print("Model saved as waste_model.pth")