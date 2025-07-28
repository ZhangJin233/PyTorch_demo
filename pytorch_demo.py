import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import mlflow
import os
from datetime import datetime

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# 定义超参数
batch_size = 128
learning_rate = 0.001
epochs = 10

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # 计算准确率
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

            # 记录训练批次级别指标
            mlflow.log_metric(
                "batch_loss", loss_val, step=batch + epoch * len(dataloader)
            )

    # 记录每个epoch的平均指标
    train_loss /= len(dataloader)
    train_accuracy = correct / size
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

    return train_loss, train_accuracy


def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    test_accuracy = correct / size

    print(
        f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    # 记录测试指标
    mlflow.log_metric("test_loss", test_loss, step=epoch)
    mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

    return test_loss, test_accuracy


# 设置MLflow实验
experiment_name = "fashion_mnist_classification"
mlflow.set_experiment(experiment_name)

# 使用时间戳创建唯一的模型路径
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f"model-{timestamp}.pth"

# 开始MLflow运行
with mlflow.start_run(run_name=f"run-{timestamp}"):
    # 记录超参数
    mlflow.log_params(
        {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "optimizer": type(optimizer).__name__,
            "model_architecture": "NeuralNetwork",
            "device": device,
        }
    )

    # 训练模型
    best_accuracy = 0.0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, t)
        test_loss, test_acc = test(test_dataloader, model, loss_fn, t)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

            # 记录最佳模型到MLflow
            mlflow.log_artifact(model_path, "models")

    # 记录最终指标
    mlflow.log_metric("best_test_accuracy", best_accuracy)
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print("Done!")

    # 记录模型
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    mlflow.log_artifact("model.pth")

    # 记录模型摘要为文本文件
    with open("model_summary.txt", "w") as f:
        f.write(str(model))
    mlflow.log_artifact("model_summary.txt")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
