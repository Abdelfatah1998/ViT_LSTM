import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# Check for CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# ViT-LSTM Model Definition
class ViTLSTM(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, lstm_layers=1, bidirectional=True):
        super(ViTLSTM, self).__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.lstm = nn.LSTM(768, hidden_dim, num_layers=lstm_layers, bidirectional=bidirectional, batch_first=True)
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a sequence dimension
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.vit(x)
        features = features.view(batch_size, seq_len, -1)
        _, (hidden, _) = self.lstm(features)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        output = self.classifier(hidden)
        return output

# Data Loading Function
def load_data(train_dir, test_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

# Evaluation Function
def evaluate(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc='Evaluating', leave=False):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    precision, recall, f1, accuracy = calculate_metrics(np.array(y_true), np.array(y_pred))
    return precision, recall, f1, accuracy

# Main Execution Function
def main():
    train_dir = r'C:\Users\u16104773\Desktop\Train-wav'
    test_dir = r'C:\Users\u16104773\Desktop\Test-wav'
    batch_size = 32
    num_classes = 24

    train_loader, test_loader = load_data(train_dir, test_dir, batch_size)

    model = ViTLSTM(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 300
    for epoch in range(epochs):
        model.train()
        train_y_true, train_y_pred, train_loss = [], [], []
        loop = tqdm(train_loader, leave=True, position=0)
        for data, targets in loop:
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_y_true.extend(targets.cpu().numpy())
            train_y_pred.extend(predicted.cpu().numpy())
            train_loss.append(loss.item())

            loop.set_postfix(loss=np.mean(train_loss))

        train_metrics = calculate_metrics(np.array(train_y_true), np.array(train_y_pred))
        print(f'Epoch {epoch+1}, Training - Loss: {np.mean(train_loss):.4f}, Accuracy: {train_metrics[3]:.4f}, Precision: {train_metrics[0]:.4f}, Recall: {train_metrics[1]:.4f}, F1-score: {train_metrics[2]:.4f}')

        test_metrics = evaluate(model, test_loader, device)
        print(f'Epoch {epoch+1}, Testing - Accuracy: {test_metrics[3]:.4f}, Precision: {test_metrics[0]:.4f}, Recall: {test_metrics[1]:.4f}, F1-score: {test_metrics[2]:.4f}')

if __name__ == "__main__":
    main()
