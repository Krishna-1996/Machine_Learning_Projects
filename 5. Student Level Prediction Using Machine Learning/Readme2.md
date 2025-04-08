# **Student Level Prediction Using Machine Learning**

When a student changes their curriculum, everything—from their study environment to teaching styles and school culture—can shift dramatically. These changes affect not only the student but also their parents, guardians, and teachers. Traditionally, discussions and meetings between parents and teachers help decide the best grade level for a student, but this approach is outdated. What’s needed is a more effective way to predict whether a student is ready for the new grade.

This project helps uncover hidden patterns in academic data and predicts which grade level is best for a student when changing schools using **Machine Learning (ML)**. It emphasizes **Ensemble Models**, **Graph-Based Methods**, and **Explainable AI (XAI)** using **LIME**. A web application was also developed to bring this solution closer to real users—students, parents, and educators—making the model actionable and interactive.

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

This project focuses on predicting if a student should move to a particular grade level based on their academic performance. Models classify students' outcomes as **pass (1)** or **fail (0)** using indicators like **study hours**, **attendance**, **subject-wise scores**, and **previous performance**.

Machine learning models evaluated:  
**SVM**, **Random Forest**, **AdaBoost**, **XGBoost**, and a **Voting Classifier**.

To make model behavior transparent, **LIME** is used to explain individual predictions, especially helpful for stakeholders with limited ML knowledge.

---

## **Models Overview**

### 1. **Support Vector Machine (SVM)**
- **Approach**: Finds the optimal hyperplane for classification.
- **Why SVM?** It showed excellent performance, especially with **numerical features** like subject scores.
- **Performance**: Achieved 98.9% accuracy. Ideal for datasets with clear margins between classes.

### 2. **Random Forest**
- **Approach**: Ensemble of decision trees using majority voting.
- **Performance**: Good accuracy and interpretability, but prone to overfitting without tuning.

### 3. **AdaBoost**
- **Approach**: Boosts weak learners by focusing on errors.
- **Performance**: Works well with imbalanced datasets.

### 4. **XGBoost**
- **Approach**: Gradient boosting with regularization.
- **Performance**: Fast and accurate with good generalization.

### 5. **Voting Classifier**
- **Approach**: Combines predictions from multiple models.
- **Performance**: Balanced results by leveraging different model strengths.

---

## **Dataset and Preprocessing**

The dataset includes features like:
- Previous grades
- Subject-wise performance
- Entrance exam results
- Curriculum history

### **Preprocessing Steps:**
- Handled missing values
- Normalized numeric features
- Encoded categorical data

**Training Set**: 594 (fail) / 524 (pass)  
**Test Set**: 148 (fail) / 131 (pass)

---

## **Evaluation Results**

| Model                | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|----------------------|----------|-----------|--------|----------|---------|
| **SVM**               | 98.9%    | 98.8%     | 98.9%  | 98.9%    | 1.000   |
| **Random Forest**     | 85.0%    | 90.1%     | 76.5%  | 82.7%    | 0.949   |
| **AdaBoost**          | 87.0%    | 88.2%     | 83.8%  | 85.8%    | 0.952   |
| **XGBoost**           | 89.3%    | 90.6%     | 86.1%  | 88.3%    | 0.966   |
| **Voting Classifier** | 92.8%    | 96.2%     | 88.1%  | 91.9%    | 0.991   |

**Confusion Matrix**  
![Confusion Matrix](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/confusion%20metrics%20output.png)

---

## **SVM Insights and LIME**

After testing all models, **SVM** stood out for its superior performance on numeric-heavy features like **Maths**, **Science**, and **English** scores.

LIME was used to explain how individual predictions were made by the SVM model, showing which features contributed the most.

**Here is the LIME explanation for SVM**  
![LIME Explanation](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/LIME_for_SVM.png)

---

## **Web Application**

To make the project accessible and useful to end-users, a web application was developed using the **SVM model**.

- The app allows users (students, teachers, parents) to input student data.
- Based on the input, the model predicts if the chosen grade level is suitable.
- It uses **LIME** to explain why the prediction was made.

**500 Random Data Points Tested**
- **321** predicted "grade appropriate"
- **179** flagged potential issues
- Confirms model consistency and usefulness in real-world applications
 
![Random Dataset Graph](https://github.com/Krishna-1996/Machine_Learning_Projects/blob/main/5.%20Student%20Level%20Prediction%20Using%20Machine%20Learning/500%20new%20dataset.png)

**Fig: 500 Random Dataset Result Visualization** 

### **WebApp Screenshots**

**Fig: User Input Page**  
![WebApp Input Page](https://your-screenshot-link-1)**Fig: User Input Page** ![WebApp Input Page](https://your-screenshot-link-1)

**Fig: Result Output Page**  
![WebApp Result Page](https://your-screenshot-link-2)



> _Note: Replace the above image URLs with links to your uploaded GitHub screenshots._

---

## **Discussion and Future Work**

### **Discussion**
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
