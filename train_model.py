import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from wasteClassifier.models import WasteClassifier  #return render(request, 'home.html', {'error': 'Uploaded file is not an image. Please upload a valid image.'})return render(request, 'home.html', {'error': 'Uploaded file is not an image. Please upload a valid image.'})return render(request, 'home.html', {'error': 'Uploaded file is not an image. Please upload a valid image.'})Import your AI model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Step 1: Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize image values to [-1, 1]
])

# Step 2: Load dataset (folder should be data/train with subfolders for classes)
dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Set up model, loss function, optimizer
model = WasteClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the model
print("🔄 Training started...\n")
for epoch in range(5):  # You can increase the number of epochs
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"✅ Epoch {epoch+1} - Loss: {running_loss:.4f}")

# Step 5: Save the trained model
torch.save(model.state_dict(), 'waste_model.pth')
print("🎉 Training complete! Model saved as 'waste_model.pth'")
