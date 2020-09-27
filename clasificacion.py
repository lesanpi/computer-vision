import torch
import torchvision
import numpy as np
from tqdm import tqdm
from RNN import Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = {
    'train': torchvision.datasets.MNIST('./data', train=True, download=True),
    'test': torchvision.datasets.MNIST('./data', train=False, download=True)
}

dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=64, shuffle=True, pin_memory=True),
    'test': torch.utils.data.DataLoader(dataset['test'], batch_size=64, shuffle=False, pin_memory=True),

}


def fit(dataloader, epochs=5, model=classifier):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters, lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # Coloca el modelo en mode de entrenamiento
        model.train()
        train_loss, train_acc = [], []
        bar = tqdm(dataloader['train'])
        for batch in bar:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            acc = (y == torch.argmax(y_pred, axis=1)).sum().item() / len(y)
            train_acc.append(acc)

            bar.set_description(
                f"Loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")

        bar = tqdm(dataloader['test'])
        val_loss, val_acc = [], []
        # Modo evaluacion
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y = batch
                X, y = X.to(device), y.to(device)

                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_pred, axis=1)).sum().item() / len(y)
                train_acc.append(acc)

                bar.set_description(
                    f"Val Loss {np.mean(train_loss):.5f} Val acc {np.mean(train_acc):.5f}")

        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {val_loss:.5f} acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f}")


classifier = Classifier()
fit(dataloader=dataloader, model=classifier)
