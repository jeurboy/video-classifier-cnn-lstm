import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tensorflow import keras
import cv2
import numpy as np


class VideoDataset(Dataset):
    def __init__(self, video_list, labels, transform=None):
        self.video_list = video_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        frames = self._load_frames(video_path)
        label = self.labels[idx]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        return frames, label

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

# Define a CNN-LSTM model


class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = models.resnet18(pretrained=True)

        # Remove the last FC layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=128,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        cnn_out = torch.zeros(batch_size, seq_len, 512).to(x.device)
        for t in range(seq_len):
            cnn_out[:, t, :] = self.cnn(x[:, t, :, :, :]).view(batch_size, -1)
        lstm_out, _ = self.lstm(cnn_out)
        out = self.fc(lstm_out[:, -1, :])
        return out


# Hyperparameters and data loaders
num_classes = 10  # Example: 10 classes
batch_size = 8
learning_rate = 0.001
num_epochs = 20


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["tag"]))
# print(label_processor.get_vocabulary())

# labels = train_df["tag"].values
# labels = label_processor(labels[..., None]).numpy()
# print(labels)

video_list = train_df["video_name"].values.tolist()
print(video_list)

# # Example video list and labels
video_list = ['dataset/train/normal/2024-05-14_13-06-43.mp4',
              'dataset/train/normal/2024-05-14_13-43-36.mp4',
              ]  # Add more video paths
labels = [0, 1]  # Corresponding labels for the videos

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = VideoDataset(video_list, labels, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = CNNLSTM(num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (videos, labels) in enumerate(dataloader):
        print("epoch start...")
        videos, labels = videos.cuda(), labels.cuda()
        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')


# Save the trained model
torch.save(model.state_dict(), 'cnn_lstm_model.pth')
print("Model saved to cnn_lstm_model.pth")
