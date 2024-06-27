import torch
import torch.nn as nn
import torch.optim as optim
from os.path import exists



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
    



def train(model_path, trainloader_list=[], num_epochs=0):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Initialize the model, loss function, and optimizer
    model = DigitModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # load the trained model
    if exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.train()

    # Training the model
    for epoch in range(num_epochs):
        for train_loader in trainloader_list:
            running_loss = 0.0
            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device) 
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {(epoch+1)}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print('Model trained and saved as digit_model.pth')




def test(model_path, testloader_list=[]):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load the pre-trained model
    model = DigitModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for test_loader in testloader_list:
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device) 
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))