import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torchvision
import sys
import os
import random
sys.path.append("./training")
from training.RNN import Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = {
    'train': torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor()),
    'test': torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
}

model = Classifier()
model.load_state_dict(torch.load("./models/classifier_state_dict.pt"))
model.eval()

r, c = 3, 5
fig = plt.figure(figsize=(2*c, 2*r))
for _r in range(r):
    for _c in range(c):
        plt.subplot(r, c, _r*c + _c + 1)
        ix = random.randint(0, len(dataset['test'])-1)
        img, label = dataset['test'][ix]
        preds = model(img.unsqueeze(0).to(device))
        pred = torch.argmax(preds, axis=1)[0].item()
        plt.imshow(img.squeeze(0), cmap='gray')
        plt.axis("off")
        plt.title(f'{label}/{pred}', color="red" if label != pred else "green")
plt.tight_layout()
plt.show()
