# **Student Level Prediction Using Machine Learning**

When a student changes their curriculum, everything—from their study environment to teaching styles and school culture—can shift dramatically. These changes affect not only the student but also their parents, guardians, and teachers. Traditionally, discussions and meetings between parents and teachers help decide the best grade level for a student, but this approach is outdated. What’s needed is a more effective way to predict whether a student is ready for the new grade. 

This project aims to help students, teachers, and parents by uncovering hidden patterns in academic data and predicting which grade level is the best fit for a student when changing schools. By using **Machine Learning (ML)**, especially **Ensemble** and **Graph-Based Methods**, this study analyzes factors like student demographics, academic performance, and engagement. Additionally, the project explores **Explainable AI (XAI)** using **LIME** (Local Interpretable Model-agnostic Explanations) to provide clear insights into the predictions. The goal is to improve prediction accuracy, enhance learning strategies, and contribute to the growing use of AI in education.
---

## **Table of Contents**

- **[Introduction](#introduction)**
- **[Models Overview](#models-overview)**
- **[Dataset and Preprocessing](#dataset-and-preprocessing)**
- **[Evaluation Results](#evaluation-results)**
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

---

## **Discussion and Future Work**

### **Future Work**
- **Hyperparameter Tuning**: Perform grid search or random search to optimize the hyperparameters of models.
- **Cross-validation**: Implement **k-fold cross-validation** to get a better estimate of model performance.
- **Deep Learning Models**: Explore neural networks like **MLP** to potentially improve predictions.

### **Discussion**
- The results show that **SVM** and **Voting Classifier** perform exceptionally well in terms of accuracy and precision.  
- **Random Forest** and **AdaBoost** are good alternatives, providing a balance between accuracy and computational efficiency.  
- The **Voting Classifier** benefits from combining multiple models, giving it a strong performance across all metrics.

---

## **Conclusion**

In conclusion, the **SVM** and **Voting Classifier** are the best-performing models for this problem. These models offer high accuracy and strong predictive power. However, different models may be preferred based on specific use cases:

- **SVM** is ideal for high-dimensional and complex datasets where accuracy is the top priority.
- **Voting Classifier** is great for combining the strengths of different models and enhancing overall performance.
- **Random Forest** and **AdaBoost** can be used for robust and well-rounded results in various scenarios.

This project demonstrates the importance of selecting the right model based on the data at hand, and further improvements can be made through hyperparameter tuning and cross-validation.

---

**Happy Predicting!** 🚀
