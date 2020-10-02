import matplotlib
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects

device = "cuda" if torch.cuda.is_available() else "cpu"

train = torchvision.datasets.VOCDetection("../data", download=True)

classes = ["background",
           "aeroplane",
           "bicycle",
           "bird",
           "boat",
           "bottle",
           "bus",
           "car",
           "cat",
           "chair",
           "cow",
           "diningtable",
           "dog",
           "horse",
           "motorbike",
           "person",
           "pottedplant",
           "sheep",
           "sofa",
           "train",
           "tvmonitor"]

# Example
img, label = train[0]
