import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import pandas as pd
import numpy as np

# Custom dataset class for loading video frames


class VideoDataset(Dataset):
    def __init__(self, video_list, transform=None):
        self.video_list = video_list
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        frames = self._load_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)
        return frames

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


# Hyperparameters
num_classes = 10  # Example: 10 classes

# Load the trained model
model = CNNLSTM(num_classes).cuda()
model.load_state_dict(torch.load('cnn_lstm_model.pth'))
model.eval()

# Example video list for prediction
video_list = ['path/to/video3.mp4',
              'path/to/video4.mp4']  # Add more video paths

# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = VideoDataset(video_list, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Prediction loop
with torch.no_grad():
    for videos in dataloader:
        videos = videos.cuda()
        outputs = model(videos)
        _, predicted = torch.max(outputs, 1)
        print(f'Predicted class: {predicted.item()}')
