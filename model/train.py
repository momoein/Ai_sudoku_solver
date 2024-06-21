import torch
import torch.nn as nn
import torch.optim as optim
from os.path import exists

from model.loader import data_loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


trainloader = data_loader["my_dataset"]["train"]

# Initialize the model, loss function, and optimizer
model = DigitModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_path = "./digit_model.pth"

if __name__ == "__main__":
    print("Device:", device)
    
    # load the trained model
    if exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.train()

    # Training the model
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}')

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print('Model trained and saved as digit_model.pth')

