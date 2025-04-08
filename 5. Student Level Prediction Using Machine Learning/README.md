# **Student Level Prediction Using Machine Learning**

When a student changes their curriculum, everything—from their study environment to teaching styles and school culture—can shift dramatically. These changes affect not only the student but also their parents, guardians, and teachers. Traditionally, discussions and meetings between parents and teachers help decide the best grade level for a student, but this approach is outdated. What’s needed is a more effective way to predict whether a student is ready for the new grade. 

This project aims to help students, teachers, and parents by uncovering hidden patterns in academic data and predicting which grade level is the best fit for a student when changing schools. By using **Machine Learning (ML)**, especially **Ensemble** and **Graph-Based Methods**, this study analyzes factors like student demographics, academic performance, and engagement. Additionally, the project explores **Explainable AI (XAI)** using **LIME** (Local Interpretable Model-agnostic Explanations) to provide clear insights into the predictions. The goal is to improve prediction accuracy, enhance learning strategies, and contribute to the growing use of AI in education.
---

## **Table of Contents**

- **[Introduction](#introduction)**
- **[Models Overview](#models-overview)**
- **[Dataset and Preprocessing](#dataset-and-preprocessing)**
- **[Evaluation Results](#evaluation-results)**
- **[SVM Insights and LIME](#svm-insights-and-lime)**
- **[Web Application](#web-application)**
- **[Discussion and Future Work](#discussion-and-future-work)**
- **[Conclusion](#conclusion)**

---

## **Introduction**

The goal of this project is to develop predictive models that classify students' academic outcomes as pass (class 1) or fail (class 0), providing insights into whether the chosen grade level is suitable for the student. The predictions are based on key performance indicators like **study hours**, **attendance**, and **previous grades**. We utilize several machine learning algorithms including **SVM**, **Random Forest**, **AdaBoost**, **XGBoost**, and **Voting Classifier** to build and evaluate these models.

The models are trained on a dataset with features influencing student performance, and evaluated using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **ROC AUC**. In addition to prediction, we incorporate **Explainable AI (XAI)** with **LIME (Local Interpretable Model-agnostic Explanations)** to interpret model decisions, offering greater transparency and insights into the prediction process. This enables data-driven educational strategies and early identification of at-risk students, ultimately improving personalized learning approaches.

### **Performance Metrics Evaluated:**

- **Accuracy**: The proportion of correct predictions out of all predictions.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of actual positive cases that were correctly identified.
- **F1-Score**: The balance between precision and recall.
- **ROC AUC**: The area under the ROC curve, representing model performance.

---

## **Models Overview**

### 1. **Support Vector Machine (SVM)**  
- **Approach**: Finds the optimal hyperplane to classify data, providing high accuracy and strong generalization.
- **Performance**: Very effective, especially in high-dimensional data with excellent precision and recall.

### 2. **Random Forest**  
- **Approach**: An ensemble method of decision trees that uses majority voting to classify the data.
- **Performance**: Generally robust with good accuracy, but can be prone to overfitting.

### 3. **AdaBoost**  
- **Approach**: A boosting method that focuses on weak learners by iteratively adjusting weights on misclassified data.
- **Performance**: Performs well on imbalanced datasets with high recall and precision.

### 4. **XGBoost**  
- **Approach**: A highly efficient gradient boosting method with regularization to prevent over-fitting.
- **Performance**: Known for fast computation and superior accuracy.

### 5. **Voting Classifier**  
- **Approach**: Combines the predictions of multiple models (such as SVM, Random Forest, AdaBoost) using majority voting.
- **Performance**: Often provides the best results by combining the strengths of different models.

---

## **Dataset and Preprocessing**

The dataset consists of student performance data, including features like **previous grades**, **entrance exam result**, and **previous curriculum**. The dataset is preprocessed with steps such as **handling missing values**, **normalizing continuous features**, and **encoding categorical variables**.

- **Training Data**: Includes 594 instances of class 0 (fail) and 524 instances of class 1 (pass).
- **Test Data**: Includes 148 instances of class 0 (fail) and 131 instances of class 1 (pass).

---

## **Evaluation Results**

Below are the evaluation results for each model, showing their **accuracy**, **precision**, **recall**, **F1-score**, and **ROC AUC** scores:

| Model                | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|----------------------|----------|-----------|--------|----------|---------|
| **SVM**               | 98.9%    | 98.8%     | 98.9%  | 98.9%    | 1.000   |
| **Random Forest**     | 85.0%    | 90.1%     | 76.5%  | 82.7%    | 0.949   |
| **AdaBoost**          | 87.0%    | 88.2%     | 83.8%  | 85.8%    | 0.952   |
| **XGBoost**           | 89.3%    | 90.6%     | 86.1%  | 88.3%    | 0.966   |
| **Voting Classifier** | 92.8%    | 96.2%     | 88.1%  | 91.9%    | 0.991   |

**The Confusion Metrics Result**
![My Image](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/confusion%20metrics%20output.png)

**The Expandable AI: LIME**
This study used LIME to explain how the model is predicting the value and what features will impact the this prediction. 
Here the LIME is representing the SVM model for instance(row) 218.

**Here is the LIME explanation for SVM**
![My Image](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/LIME_for_SVM.png)

---
---

## **Web Application**

To make the project accessible and useful to end-users, a web application was developed using the **SVM model**.

- The app allows users (students, teachers, parents) to input student data.
- Based on the input, the model predicts if the chosen grade level is suitable.
- It uses **LIME** to explain why the prediction was made.

**500 Random Data Points Tested**
- **321** predicted "grade appropriate"
- **179** flagged potential issues
- Confirms model consistency and usefulness in real-world applications.

**500 Random Dataset Result Visualization**<br>
![Random Dataset Graph](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/500%20new%20dataset.png)

---

### **WebApp Screenshots**

**User Input Page**<br>
![WebApp Input Page ](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/Webpage%20front%20page1.png)<br>


**User Submit Page**<br>
![WebApp Submit Page ](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/Webpage%20front%20page2.png)<br>

**Result Page** <br> 
![WebApp Result Page ](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/Webpage%20result%20page.png)

---

## **Discussion and Future Work**

- **SVM** and **Voting Classifier** outperformed others across all metrics.
- SVM’s reliability with numeric features makes it ideal for this context.
- The **web app** bridges model utility and real-world accessibility.
- **LIME** adds transparency, building user trust in predictions.

### **Future Work**
- **Hyperparameter Tuning**: Use grid or random search for fine-tuning models.
- **Explore Deep Learning**: Test MLP or other deep models for potential improvements.

---

## **Conclusion**

This project successfully demonstrates how ML can assist academic decision-making:

- **SVM** is the top performer for predicting student readiness.
- **WebApp** makes predictions accessible and actionable.
- **Explainability** via LIME increases model interpretability.

> Students, parents, and educators can now leverage this tool for informed, data-driven educational planning.

---

**Happy Predicting!** 🚀
