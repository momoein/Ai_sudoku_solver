import torch
import torch.nn as nn

from .train import DigitModel
from model.loader import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

testloader = data_loader["my_dataset"]["test"]

# Load the pre-trained model
model_path = "./digit_model.pth"
model = DigitModel().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device) 
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))