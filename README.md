# PyTorch_demo

1. 环4. 实验与分析
- 在项目目录下运行 mlflow ui。
- 现在，你的脚本已经准备好了。保持你的 mlflow ui 窗口开着，然后打开一个新的终端（确保进入了正确的虚拟环境和项目目录），开始你的"参数测试".
- 进行对比实验：修改超参数（例如，尝试不同的学习率 0.01, 0.001），多次运行你的脚本。![alt text](image-1.png)
- 在MLflow UI中分析：
    - 对比不同"Run"的参数和最终指标，找出效果最好的那次实验。
    - 点击进入某次"Run"的详情页，查看Metrics随时间变化的图表。
    - 在"Artifacts"中，直接预览你保存的混淆矩阵图和分类报告。
- 核心体验：感受MLflow如何让你的每次模型"测试"都变得有据可查、可追溯、可对比。

5. 独立评估脚本
- 项目中包含一个独立的评估脚本 `evaluate.py`，可以从MLflow加载训练好的模型并进行评估。
- 使用方法：
  ```bash
  python evaluate.py --run-id <RUN_ID> --batch-size 64 --log-mlflow
  ```
  其中：
  - `<RUN_ID>` 是MLflow实验运行的ID，可以从MLflow UI中获取
  - `--batch-size` 可选参数，指定评估时的批次大小
  - `--log-mlflow` 可选参数，如果提供则将评估结果记录回MLflow
- 评估脚本会生成:
  - 详细的分类报告（classification_report.json）
  - 混淆矩阵可视化（confusion_matrix.png）
  - 错误分类样本可视化（misclassified_examples.png）
- 这种训练与评估解耦的方法是自动化测试的基础，便于CI/CD集成和模型的持续评估。个项目文件夹，使用venv或conda创建虚拟环境。
- 安装必要的库：pip install torch torchvision numpy pandas scikit-learn mlflow。
2. 编写基础训练脚本 (不含MLflow)
- 加载torchvision自带的MNIST数据集。
- 定义一个简单的卷积神经网络（CNN）模型（可以直接从PyTorch官方教程复制）。
- 编写训练和评估函数。
- 在训练结束后，在测试集上进行评估，并手动打印混淆矩阵和分类报告（使用scikit-learn）。
- 目标：先跑通一个能正常工作的基线版本（Baseline）。
3. 集成MLflow Tracking
- 改造脚本：在训练脚本中引入mlflow。
- mlflow.start_run()：将你的训练和评估代码包裹起来。
- mlflow.log_param()：记录本次实验的超参数，例如：学习率（learning rate）、批量大小（batch size）、训练轮次（epochs）。
- mlflow.log_metric()：在每个epoch结束时，记录训练损失（loss）和验证准确率（validation accuracy）。在最终测试后，记录测试集上的所有关键指标（Accuracy, Precision, Recall, F1-Score）。
- mlflow.log_artifact()：记录输出产物。
- 将scikit-learn生成的混淆矩阵图保存为图片（e.g., confusion_matrix.png），然后记录它。
    - 将完整的分类报告保存为文本文件（e.g., classification_report.txt），然后记录它。
    - mlflow.pytorch.log_model()：将训练好的PyTorch模型（.pth文件）作为一个MLflow模型记录下来。
4. 实验与分析
- 在项目目录下运行 mlflow ui。
- 现在，你的脚本已经准备好了。保持你的 mlflow ui 窗口开着，然后打开一个新的终端（确保进入了正确的虚拟环境和项目目录），开始你的“参数测试”.
- 进行对比实验：修改超参数（例如，尝试不同的学习率 0.01, 0.001），多次运行你的脚本。![alt text](image-1.png)
- 在MLflow UI中分析：
    - 对比不同“Run”的参数和最终指标，找出效果最好的那次实验。
    - 点击进入某次“Run”的详情页，查看Metrics随时间变化的图表。
    - 在"Artifacts"中，直接预览你保存的混淆矩阵图和分类报告。
- 核心体验：感受MLflow如何让你的每次模型“测试”都变得有据可查、可追溯、可对比。

5. 一个简单的CI流水线（以GitHub Actions为例）： 
- 触发器：当代码合并到main分支时触发。 
- 执行训练：运行pytorch_demo.py 脚本，所有结果自动记录到MLflow Server。 
- 执行评估：运行evaluate.py脚本，加载刚刚训练出的模型。
- 质量门禁（Quality Gate）：脚本判断模型的关键指标（如Accuracy）是否大于预设阈值（例如0.98）。 
- 自动晋级：如果通过门禁，则通过API调用，将MLflow Registry中的模型版本从“Staging”更新为“Production”。如果失败，则流水线报错。 
- 测试要点：这就是将模型评估融入到CI/CD中，实现了AI质量的自动化监控。