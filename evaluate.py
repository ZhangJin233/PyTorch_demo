import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import mlflow
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


# 定义相同的神经网络模型结构以确保兼容性
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


def load_model_from_mlflow(run_id):
    """
    从MLflow加载指定run_id的模型
    """
    print(f"正在从MLflow加载run_id为 {run_id} 的模型...")

    # 获取运行的信息
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    # 从运行信息中获取参数
    params = run.data.params
    print(f"模型参数: {params}")

    # 获取模型文件路径
    artifacts_path = client.download_artifacts(run_id, "model.pth")
    print(f"模型文件已下载到: {artifacts_path}")

    # 创建并加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    model.load_state_dict(
        torch.load(artifacts_path, map_location=device, weights_only=True)
    )
    model.eval()

    return model, device, params


def load_test_data(batch_size=64):
    """
    加载测试数据
    """
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return test_dataloader, test_data


def evaluate_model(model, test_dataloader, device):
    """
    评估模型并返回结果
    """
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    correct = 0

    # 保存所有预测和真实标签以便进行详细分析
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    accuracy = correct / size

    return {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def generate_classification_report(predictions, labels, class_names):
    """
    生成并保存分类报告
    """
    # 计算分类报告
    report = classification_report(
        labels, predictions, target_names=class_names, output_dict=True
    )

    # 打印文本形式的分类报告
    print("\n分类报告:")
    print(classification_report(labels, predictions, target_names=class_names))

    # 将报告保存为JSON
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("分类报告已保存到 classification_report.json")

    return report


def plot_confusion_matrix(predictions, labels, class_names):
    """
    绘制并保存混淆矩阵
    """
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵")
    plt.tight_layout()

    # 保存图表
    plt.savefig("confusion_matrix.png")
    print("混淆矩阵已保存到 confusion_matrix.png")

    return cm


def plot_misclassified_examples(
    model, test_data, predictions, labels, class_names, device, num_examples=10
):
    """
    可视化错误分类的示例
    """
    misclassified_indices = np.where(predictions != labels)[0]

    if len(misclassified_indices) == 0:
        print("没有错误分类的示例！")
        return

    # 选择指定数量的错误分类示例
    samples = min(num_examples, len(misclassified_indices))
    indices = np.random.choice(misclassified_indices, samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        img, label = test_data[idx]
        img = img.to(device)

        with torch.no_grad():
            pred = model(img.unsqueeze(0))
            predicted_label = pred.argmax(1).item()

        axes[i].imshow(img.cpu().squeeze(), cmap="gray")
        axes[i].set_title(
            f"真实: {class_names[label]}\n预测: {class_names[predicted_label]}"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("misclassified_examples.png")
    print("错误分类示例已保存到 misclassified_examples.png")


def log_evaluation_to_mlflow(run_id, metrics, artifacts_dir="."):
    """
    将评估结果记录到MLflow中的同一个run
    """
    with mlflow.start_run(run_id=run_id):
        # 记录评估指标
        mlflow.log_metric("eval_test_loss", metrics["test_loss"])
        mlflow.log_metric("eval_accuracy", metrics["accuracy"])

        # 记录生成的工件
        artifact_files = [
            "classification_report.json",
            "confusion_matrix.png",
            "misclassified_examples.png",
        ]

        for file in artifact_files:
            if os.path.exists(os.path.join(artifacts_dir, file)):
                mlflow.log_artifact(os.path.join(artifacts_dir, file), "evaluation")

    print(f"评估结果已记录到MLflow运行ID: {run_id}")


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="从MLflow加载模型并进行评估")
    parser.add_argument("--run-id", type=str, required=True, help="MLflow运行ID")
    parser.add_argument("--batch-size", type=int, default=64, help="评估的批次大小")
    parser.add_argument(
        "--log-mlflow", action="store_true", help="是否将评估结果记录到MLflow"
    )

    args = parser.parse_args()

    # FashionMNIST数据集的类别名称
    class_names = [
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

    # 加载模型
    model, device, params = load_model_from_mlflow(args.run_id)

    # 加载测试数据
    test_dataloader, test_data = load_test_data(batch_size=args.batch_size)

    # 评估模型
    print("正在评估模型...")
    evaluation_results = evaluate_model(model, test_dataloader, device)

    # 输出主要指标
    print(f"\n测试损失: {evaluation_results['test_loss']:.6f}")
    print(
        f"准确率: {evaluation_results['accuracy']:.6f} ({evaluation_results['accuracy']*100:.2f}%)"
    )

    # 生成分类报告
    report = generate_classification_report(
        evaluation_results["predictions"], evaluation_results["labels"], class_names
    )

    # 绘制混淆矩阵
    cm = plot_confusion_matrix(
        evaluation_results["predictions"], evaluation_results["labels"], class_names
    )

    # 可视化错误分类的示例
    plot_misclassified_examples(
        model,
        test_data,
        evaluation_results["predictions"],
        evaluation_results["labels"],
        class_names,
        device,
    )

    # 如果指定，将评估结果记录到MLflow
    if args.log_mlflow:
        log_evaluation_to_mlflow(args.run_id, evaluation_results)

    print("\n评估完成!")


if __name__ == "__main__":
    main()
