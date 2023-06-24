# 增加输入，eca模块

import math
import torch
import torch.nn as nn
from torchvision import models


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v


class AutoDriveModel(nn.Module):
    def __init__(self):
        super(AutoDriveModel, self).__init__()

        # Load pretrained ResNet
        self.resnet = models.regnet_y_400mf(pretrained=True)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # LSTM layer: 512 as the last convolutional layer of pretrained ResNet-18 has 512 output channels
        self.lstm_img = nn.LSTM(1000, 256, batch_first=True)




        # For Timing speed and steer, it should only have one layer. 256 to keep it consistent with the hidden state size of the LSTM network processing images.
        self.lstm_speed = nn.LSTM(1, 256, batch_first=True)
        self.lstm_steer = nn.LSTM(1, 256, batch_first=True)

        # Fully connected layer
        self.fc_speed = nn.Linear(256, 256)
        self.fc_steer = nn.Linear(256, 256)
        self.speed_output = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 20),
        )
        self.steer_output = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 20),
        )
        
        # ECA layer
        self.eca = ECABlock(768)

    def forward(self, img_sequence, speed_sequence, steer_sequence):
        # Extract image features with ResNet
        img_features = [self.resnet(img) for img in img_sequence]
        img_features = torch.stack(img_features).squeeze()
        # print('img_features.shape: ', img_features.shape)

        # LSTM processing
        img_output, _ = self.lstm_img(img_features)
        # print('img_lstm_output.shape: ', img_output.shape)
        speed_output, _ = self.lstm_speed(speed_sequence)
        steer_output, _ = self.lstm_steer(steer_sequence)

        # Fully connected layer
        speed_features = self.fc_speed(speed_output[:, -1, :])
        steer_features = self.fc_steer(steer_output[:, -1, :])
        fused_features = torch.cat((img_output[:, -1, :], speed_features, steer_features), dim=1)


        # ECA block
        fused_features = self.eca(fused_features)
        fused_features = fused_features.squeeze()
        speed_output = self.speed_output(fused_features)
        steer_output = self.steer_output(fused_features)

        return speed_output, steer_output


if __name__ == '__main__':
    # Assuming we have a batch of 5 samples, each with 10 frames of images
    batch_size = 2
    num_frames = 4
    num_channels = 3
    img_height = 224
    img_width = 224

    img_sequence = torch.randn((batch_size, num_frames, num_channels, img_height, img_width))
    speed_sequence = torch.randn((batch_size, num_frames, 1))
    steer_sequence = torch.randn((batch_size, num_frames, 1))

    model = AutoDriveModel()
    speed_output, steer_output = model(img_sequence, speed_sequence, steer_sequence)

    print('Speed output shape:', speed_output.shape)
    print('Steer output shape:', steer_output.shape)
