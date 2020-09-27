import torch
import torch.nn.functional as F


def block(c_in, c_out, kernel=3, padding=1, stride=1, pool_kernel=2, pool_stride=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=padding),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(pool_kernel, stride=pool_stride)
    )


def block2(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, c_out),
        torch.nn.ReLU()
    )


class Classifier(torch.nn.Module):
    def __init__(self, n_channels=1, n_outputs=10):
        super().__init__()
        self.conv1 = block(n_channels, 64)
        self.conv2 = block(64, 128)
        self.fc = torch.nn.Linear(128*7*7, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# Testing el Classifier
model = Classifier()
output = model(torch.randn(64, 1, 28, 28))
print(output.shape)


class Localization():
    def __init__(self):
        super().__init__()


class LocClassifier():
    def __init__(self):
        super().__init__()
