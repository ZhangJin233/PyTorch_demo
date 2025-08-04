# PyTorch_demo

## Project Overview

This project demonstrates how to use PyTorch with MLflow for model training, tracking, and evaluation. Through this project, you will learn how to use MLflow to track experiments, record metrics and parameters, and how to create an independent evaluation script to decouple the training and testing processes.

## Environment Setup

1. Create a project folder and set up a virtual environment using venv or conda
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

2. Install the necessary libraries
```bash
pip install -r requirements.txt
```

## Project Steps

### 1. Basic Training Script
- Load the FashionMNIST dataset from torchvision
- Define a simple neural network model
- Write training and evaluation functions
- Perform evaluation on the test set after training
- Goal: Get a working baseline version running first

### 2. Integrate MLflow Tracking
- Refactor the script: Introduce mlflow into your training script
- `mlflow.start_run()`: Wrap your training and evaluation code within it
- `mlflow.log_param()`: Log the hyperparameters for this experiment, such as learning rate, batch size, and epochs
- `mlflow.log_metric()`: At the end of each epoch, log the training loss and validation accuracy
- `mlflow.log_artifact()`: Log output artifacts, such as confusion matrix plot and classification report
- `mlflow.pytorch.log_model()`: Log the trained PyTorch model

### 3. Experimentation and Analysis
- Run `mlflow ui` in your project directory
- Conduct comparative experiments: Modify hyperparameters (e.g., try different learning rates) and run your script multiple times
- Analyze in the MLflow UI:
  - Compare the parameters and final metrics of different "Runs" to find the best-performing experiment
  - View charts of how metrics changed over time
  - Preview the saved confusion matrix plot and classification report
- Core Experience: Experience how MLflow makes every model "test" well-documented, traceable, and comparable

### 4. Independent Evaluation Script
- The project includes a separate evaluation script, `evaluate.py`, which can load a trained model from MLflow and evaluate it
- Usage:
  ```bash
  python evaluate.py --run-id <RUN_ID> --batch-size 64 --log-mlflow
  - A detailed classification report (`classification_report.json`)
  - A confusion matrix visualization (`confusion_matrix.png`)
  - A visualization of misclassified examples (`misclassified_examples.png`)
- This method of decoupling training and evaluation is fundamental for automated testing, facilitating CI/CD integration and continuous model assessment

### 5. CI/CD Integration Example
- Trigger: Triggered when code is merged into the main branch
- Run Training: Run the pytorch_demo.py script. All results are automatically logged to the MLflow Server
- Run Evaluation: Run the evaluate.py script, loading the model that was just trained
- Quality Gate: The script checks if a key metric of the model (e.g., Accuracy) exceeds a predefined threshold
- Automatic Promotion: If the gate is passed, promote the model version in the MLflow Registry from "Staging" to "Production"
- Key Takeaway: This is how you integrate model evaluation into CI/CD, achieving automated AI quality monitoring

## Comprehensive Analysis

### Understanding the Meaning of Core Metrics

Let's assume we are working on a multi-class classification task (like FashionMNIST classification, with 10 classes).

#### 1. Loss

* **Training/Validation Loss (`train_loss` / `validation_loss`)**:
  * **Meaning**: The "discrepancy" between the model's predictions and the true labels. The lower this value, the better the model fits the corresponding dataset.
  * **How to Evaluate**:
    * **Consistent Decrease**: In a healthy training process, both values should steadily decrease as epochs increase.
    * **Observe the Inflection Point**: When validation loss stops decreasing or even starts to rise, it usually means the model is beginning to overfit.
    * **What matters is the trend**: The absolute value of the loss itself is not very meaningful, what matters is its trend.

* **Test Loss (`test_loss`)**:
  * **Meaning**: The loss of the model on a completely unseen test set. This is the ultimate measure of the model's generalization ability.
  * **How to Evaluate**: Among multiple experiments, the one with the lowest test loss is usually one of our top candidates.

#### **2. Accuracy**

*   **`test_accuracy`**:
    *   **Meaning**: The proportion of correctly predicted samples to the total number of samples in the test set. It's the most intuitive performance metric. `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
    *   **How to Evaluate**:
        *   **Higher is Better**: Obviously, we want the accuracy to be as high as possible.
        *   **Set a Baseline**: A model has an accuracy of `98%`. Is that good or bad? You need a basis for comparison.
            *   **Random Guessing Baseline**: For a 10-class problem, random guessing has an accuracy of `10%`. `98%` is far better, indicating the model has learned something.
            *   **Business Baseline**: What is the minimum acceptable accuracy for this task in a business context? For example, anything below `95%` might be unacceptable.
            *   **State-of-the-Art (SOTA) Baseline**: On the public MNIST dataset, the best models can achieve accuracy above `99.7%`. While `98%` is good, there's still room for improvement.
        *   **Beware of Data Imbalance**: If 90% of your samples are class "1," a model that mindlessly predicts "1" for everything will achieve 90% accuracy. In this case, `Accuracy` is highly misleading. Therefore, we need more detailed metrics.

#### **3. Precision, Recall, F1-Score (usually viewed via a `classification_report`)**

These metrics help you deeply analyze the model's performance on **each class**, which is crucial, especially with imbalanced data.

Let's say we are particularly concerned with recognizing the digit "8":

#### 2. Accuracy

* **Test Accuracy (`test_accuracy`)**:
  * **Meaning**: The proportion of correctly predicted samples to the total number of samples in the test set.
  * **How to Evaluate**:
    * **Higher is Better**: Obviously, we want the accuracy to be as high as possible.
    * **Set a Baseline**: A model has an accuracy of 98%. Is that good or bad? You need a basis for comparison, such as random guessing baseline, business baseline, or state-of-the-art baseline.
    * **Beware of Data Imbalance**: When data class distribution is uneven, accuracy alone can be misleading.

#### 3. Precision, Recall, F1-Score

These metrics help you deeply analyze the model's performance on each class, which is crucial, especially with imbalanced data.

* **Precision**:
  * **Meaning**: Of all the samples the model predicted as a certain class, how many were actually that class?
  * **Formula**: `TP / (TP + FP)`
  * **Business Scenario**: In spam detection, you want high Precision. You don't want to misclassify important emails as spam.

* **Recall**:
  * **Meaning**: Of all the samples that are truly a certain class, how many did the model successfully identify?
  * **Formula**: `TP / (TP + FN)`
  * **Business Scenario**: In medical diagnostics, you want high Recall. You don't want to miss a single actual patient.

* **F1-Score**:
  * **Meaning**: The harmonic mean of Precision and Recall, providing a single, comprehensive measure.
  * **Formula**: `2 * (Precision * Recall) / (Precision + Recall)`
  * **How to Evaluate**: When you want both Precision and Recall to perform well, the F1-Score is an excellent single metric for evaluation.

---

### **Summary: A Complete Evaluation Checklist**

When you evaluate a model, you can ask yourself the following questions:

1.  **【Primary Goal】** Does the `test_accuracy` meet the business baseline? Is it the highest among all experiments?
2.  **【Generalization Ability】** Is the `test_loss` the lowest? Is the gap between `validation_loss` and `train_loss` too large (indicating overfitting)?
3.  **【Robustness/Balance】** In the `classification_report`, is there any class with a particularly low F1-Score? Is the model's performance balanced?
4.  **【Specific Errors】** What specific error patterns does the `confusion_matrix` reveal? Are these error patterns critical from a business perspective?
5.  **【Cost Consideration】** Did the model with the highest accuracy use more `epochs` (longer training time) or a more complex architecture (longer inference time)? If accuracy is comparable, choose the model with the **lower cost**.