  # Credit Card Approval Prediction Using Machine Learning
![image](https://github.com/user-attachments/assets/ea6ddcee-10a3-40dd-9a50-cf6a2404630d)


## Introduction
Predicting credit card approvals is crucial for financial institutions in assessing creditworthiness, reducing risk, and optimizing decision-making processes. This project explores a comprehensive workflow that includes:

- **Data Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Feature Engineering**
- **Model Training**
- **Model Evaluation**

We leverage machine learning models to predict credit card application approvals. The dataset, sourced from Kaggle's **Credit Card Approval Prediction Dataset**, contains demographic and socio-economic attributes, providing a well-rounded base for analysis and prediction.

## Project Workflow

1. **Data Loading and Initial Processing**  
   We start by loading and merging two datasets, `Application Record` and `Credit Record`, on the unique identifier **ID**. This combined dataset allows for a unified approach to analysis.

## Essential Libraries and Data Loading

```python
# Essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Scikit-Learn libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load and merge datasets
cf = pd.read_csv('/content/drive/MyDrive/MY projects/application_record.csv')
cr = pd.read_csv('/content/drive/MyDrive/MY projects/credit_record.csv')
cf = pd.merge(cf, cr, on='ID', how='left')
```

## 2. Exploratory Data Analysis (EDA)

### Checking for Missing Values and Data Distribution

To understand data quality, we checked for missing values and visualized the distribution of numerical features. **Histograms** helped identify skewed distributions, while **box plots** highlighted outliers.

```python
# Check for missing values
cf.isnull().sum()

# Plot histograms for numerical features
cf.select_dtypes(include=np.number).hist(figsize=(12, 10))
plt.suptitle('Distribution of Numerical Features')
plt.show()
```
### Correlation Matrix for Numerical Features

The correlation matrix is a powerful tool for examining relationships between numerical features in a dataset. Each cell represents the correlation coefficient between two features, quantifying their relationship strength. Here's an in-depth breakdown:

#### Positive Correlations
Positive values, represented by darker shades of red, indicate a direct relationship between two features. For example:
- **CNT_CHILDREN** and **CNT_FAM_MEMBERS** have a high positive correlation of 0.89. This strong correlation suggests that households with more children also tend to have more family members, which is expected and reasonable.

#### Negative Correlations
Negative values, represented by shades of blue, indicate an inverse relationship between features. For instance:
- **DAYS_BIRTH** and **DAYS_EMPLOYED** have a notable negative correlation of approximately -0.61. This suggests that older individuals (higher DAYS_BIRTH) may have fewer days of employment (DAYS_EMPLOYED), indicating that older applicants are less likely to be actively employed.

#### Near-Zero Correlations
Values close to zero indicate no linear relationship between features. For example:
- **FLAG_WORK_PHONE** and **AMT_INCOME_TOTAL** have an almost zero correlation, implying that owning a work phone does not have a linear association with income level.

#### Multicollinearity Concerns
High correlations between certain features (e.g., **CNT_CHILDREN** and **CNT_FAM_MEMBERS**) can introduce multicollinearity in the model, potentially impacting stability and interpretability in regression-based methods. To address this:
- We might consider **removing** one of these highly correlated features or **combining** them to simplify the model and avoid redundancy.

#### Impact on Feature Selection
By analyzing the correlation matrix, we can make informed decisions about feature selection:
- Highly correlated features may provide redundant information, and **dropping or combining** one can reduce model complexity without sacrificing predictive power.

In summary, the correlation matrix provides an overview of the data's internal structure, helping us understand how different attributes interact. This insight guides our feature selection and engineering steps, setting a strong foundation for building effective predictive models.
### Correlation Matrix Visualization

To visualize the correlation between numerical features, we use a heatmap. This allows us to quickly identify the strength and direction of relationships between features.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot correlation matrix heatmap
sns.heatmap(cf.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```
![image](https://github.com/user-attachments/assets/87421474-ce33-4e10-9b50-2934511af131)

### Distribution of Numerical Features

The histograms below show the distribution of each numerical feature in the dataset. Examining these distributions allows us to:

- **Identify Skewness**: Features that are highly skewed may need transformation to improve model performance.
- **Detect Outliers**: Outliers can be spotted as values that are far from the main distribution.
- **Assess Data Range**: Understanding the range of values helps in scaling or normalizing the data.

By visualizing each feature’s distribution, we can decide on further preprocessing steps, such as transformations or handling outliers.
### Code for Visualizing Distribution of Numerical Features

To visualize the distribution of each numerical feature, we use histograms. This helps in identifying skewness, outliers, and the range of values.

```python
import matplotlib.pyplot as plt

# Select numerical columns
numerical_columns = cf.select_dtypes(include=np.number).columns

# Plot histograms for numerical features
cf[numerical_columns].hist(figsize=(12, 10))
plt.suptitle('Distribution of Numerical Features', y=1.02)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/06004462-090a-42dc-b4a9-7f95692be64d)

### Box Plot of Numerical Features

The box plot below shows the distribution and presence of outliers for each numerical feature. This visualization is helpful in identifying:

- **Outliers**: Points outside the whiskers indicate outliers, which may need to be handled during data preprocessing.
- **Spread of Data**: The length of the box shows the interquartile range (IQR), indicating the spread of the middle 50% of values.
- **Feature Scaling**: Features with widely differing scales may need normalization or standardization.

By analyzing the box plot, we can identify features with significant outliers and variations in scale, helping guide our data preprocessing strategy.

```python
# Generate box plots for numerical features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
sns.boxplot(data=cf.select_dtypes(include=np.number))
plt.title('Box Plot of Numerical Features')
plt.xticks(rotation=90)
plt.show()
```

![image](https://github.com/user-attachments/assets/333cc10e-212b-45b3-b376-ae4c52d528d1)



## 3. Feature Engineering

Feature engineering was vital in this project. Here's how we crafted new features to enhance model performance and interpretability:

- **Income Categories**: Classified income into three categories (*low*, *medium*, and *high*) based on quantiles.
- **Age and Employment Duration**: Derived *age* and *employment duration* from **DAYS_BIRTH** and **DAYS_EMPLOYED**, making these features more interpretable and predictive.

### Feature Engineering Code

To implement the feature engineering steps, we derived age, employment duration, and income category features as follows:

```python
# Derive age and employment duration in years
cf['age'] = -cf['DAYS_BIRTH'] / 365
cf['employment_duration'] = -cf['DAYS_EMPLOYED'] / 365

# Classify income into categories
cf['income_category'] = cf['AMT_INCOME_TOTAL'].apply(lambda x: "low income" if x <= 25000 
                                                     else "medium income" if x <= 75000 
                                                     else "high income")
```
### Adjusted Income Categorization Function

To categorize income levels more accurately, we define a function that assigns applicants to income groups based on specific thresholds.

```python
# Define income categorization function
def Income_categories(AMT_INCOME_TOTAL):
    if AMT_INCOME_TOTAL <= 25000:
        return "low income"
    elif AMT_INCOME_TOTAL <= 75000:
        return "medium income"
    else:
        return "high income"

```

## 4. Model Selection and Training

We trained three models to predict credit approval: **Decision Tree**, **Logistic Regression**, and **Random Forest**. Each model provided unique insights, but the **Random Forest Classifier** emerged as the most effective due to its balanced accuracy, interpretability, and resistance to overfitting.

### Decision Tree Classifier
The Decision Tree model achieved high accuracy on the training data but was prone to overfitting. We controlled its complexity using parameters like `max_depth` and `min_samples_split`.

```python
from sklearn.tree import DecisionTreeClassifier

# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
dt_model.fit(X_train, y_train)
```

### Logistic Regression
While logistic regression is an effective linear model, it struggled with the non-linear patterns in the dataset, achieving only 53.57% accuracy.

```python
from sklearn.linear_model import LogisticRegression

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```

### Random Forest Classifier
The Random Forest model performed best, achieving an accuracy of **91.21%** and an AUC score of **0.96**. Its ensemble approach combines multiple decision trees to improve stability and accuracy, handling feature interactions well and making it less prone to overfitting than individual decision trees.

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=400, max_depth=30)
rf_model.fit(X_train, y_train)
```

## 5. Model Evaluation

### Confusion Matrix
The confusion matrix shows how well the model classified approvals and rejections, highlighting specific areas for improvement, such as false negatives.

- **True Positives (93,213)**: Correctly identified approvals, showing strong accuracy in approving applications.
- **True Negatives (121,999)**: Correctly identified rejections, indicating the model effectively spots applicants with lower approval likelihood.
- **False Positives (4,397)**: Incorrectly classified rejections as approvals, minimized to reduce risk.
- **False Negatives (16,354)**: Missed approvals, an area for potential model improvement.

This matrix demonstrates the model's overall reliability, with low false positives and strong performance in both approval and rejection classifications.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate confusion matrix
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.show()
```

![image](https://github.com/user-attachments/assets/a1396cad-3761-41a4-869d-adde641bc77b)

### ROC Curve and AUC Score

The Receiver Operating Characteristic (ROC) Curve is a graphical representation of a model's performance across various threshold values. It displays the trade-off between the **True Positive Rate (Sensitivity)** and **False Positive Rate**. A perfect classifier would have a point at (0,1), meaning a 100% True Positive Rate and 0% False Positive Rate.

#### AUC (Area Under Curve)
The AUC (Area Under Curve) score, which ranges between 0 and 1, summarizes the model's overall ability to distinguish between the positive and negative classes:

- **AUC = 0.5**: Model performs no better than random chance.
- **AUC = 1.0**: Perfect model.

In this case, the Random Forest model achieved an **AUC score of 0.96**, indicating excellent discriminatory ability between the classes.

#### Interpretation
- **True Positive Rate (Y-axis)**: The proportion of actual positives correctly identified by the model.
- **False Positive Rate (X-axis)**: The proportion of actual negatives incorrectly identified as positives by the model.

The closer the ROC curve is to the top-left corner, the better the model's performance. The diagonal line represents the performance of a random classifier (AUC = 0.5). Our model's ROC curve is significantly above this line, confirming that the Random Forest classifier is effective in distinguishing between the classes.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Calculate AUC score
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"Random Forest AUC Score: {auc_score:.2f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc='lower right')
plt.show()
```

![image](https://github.com/user-attachments/assets/04bc688a-2737-4979-ad6b-a6fdebbda08c)

### Feature Importance in Random Forest Model

Feature importance provides insight into which features are the most influential in the Random Forest model's decision-making process. The higher the importance score, the more significant the feature in predicting the target variable.

#### Key Features
- **DAYS_BIRTH**: This feature has the highest importance, indicating that the model relies heavily on the applicant's age for classification.
- **DAYS_EMPLOYED**: The length of employment also plays a significant role in predictions, suggesting that job stability might be a key indicator.
- **AMT_INCOME_TOTAL**: The applicant's total income contributes to the model, though to a lesser extent than age and employment duration.
- **CNT_FAM_MEMBERS**: The number of family members has the lowest importance among the top features, but it still contributes to the model.

#### Interpretation
Understanding feature importance helps in:
- **Model Interpretation**: It allows us to explain why the model makes specific predictions.
- **Feature Engineering**: We can focus on the most critical features to improve the model further or to simplify it by removing low-importance features.
- **Business Insights**: This analysis shows which applicant attributes (e.g., age, employment duration) are most relevant to the model's predictions, providing valuable insights for decision-makers.

The plot below visualizes each feature's importance, highlighting the significant predictors that contribute to the Random Forest model's performance.

```python
# Retrieve feature importances from the Random Forest model
importances = rf_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
```


![image](https://github.com/user-attachments/assets/9da3aa98-0db1-4ecf-b06c-f56382f4e5ee)


### Visualization of a Single Decision Tree in the Random Forest

The plot below displays a single decision tree from our Random Forest model. This visualization provides insight into the structure of individual trees within the ensemble. Each node in the tree represents a decision point based on one of the features. Here's a breakdown of the elements:

#### Key Elements in the Tree Visualization:
- **Nodes**: Each node contains:
  - **Feature and threshold**: The feature and threshold value used for the split.
  - **Gini index**: Measures the impurity of the node; lower values indicate purer nodes.
  - **Sample count**: Shows the number of samples in each node.
  - **Value**: Displays the distribution of classes (Rejected vs. Approved) within the node.
  - **Class**: Indicates the predicted class for the samples in that node.

- **Pathways**:
  - Left branches represent samples that meet the condition (e.g., <= threshold).
  - Right branches represent samples that do not meet the condition (> threshold).

#### Key Observations:
- **Root Node**: The tree starts with a split on one of the key features, e.g., `AMT_INCOME_TOTAL`, showing the primary decision point.
- **Depth and Complexity**: We limited the visualization to a depth of 3 to focus on the top decisions in the tree. Deeper levels add complexity and provide further fine-tuning but may lead to overfitting if examined alone.
- **Class Distribution**: The model uses features like `DAYS_EMPLOYED`, `DAYS_BIRTH`, and `AMT_INCOME_TOTAL` to separate classes, aiming to reduce impurity in each node.

### Importance of Visualizing Individual Trees

While Random Forests combine the results of many trees, examining an individual tree helps in understanding:

- **Decision Rules**: Identify the logic and thresholds that guide decisions within the model.
- **Feature Influence**: See which features are chosen as split points and how they affect the classification.
- **Interpretability**: Helps explain to stakeholders how specific features, like age or income, contribute to approval or rejection.

This visualization provides a clearer understanding of how certain factors influence decisions within one part of the Random Forest, offering insight into the decision-making process behind individual predictions.
### Code for Visualizing a Single Decision Tree in the Random Forest

The code below extracts and visualizes a single decision tree from the Random Forest model, providing insight into the decision rules and feature influences within the ensemble.

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract a single tree from the Random Forest model
single_tree = rf_model.estimators_[0]

# Set up the plot
plt.figure(figsize=(20, 10), dpi=100)  # Increase DPI for better resolution

# Plot the tree with improved readability
plot_tree(
    single_tree,
    filled=True,
    feature_names=X_train.columns,
    class_names=['Rejected', 'Approved'],
    max_depth=3,  # Adjust this to display more or less of the tree
    fontsize=10  # Adjust font size for readability
)

plt.title("Visualization of a Single Decision Tree in the Random Forest")
plt.show()
```


![image](https://github.com/user-attachments/assets/57efc759-2289-4dcc-b679-da79c8e577ee)


## Conclusion

This project successfully demonstrated the process of building a predictive model for credit card approvals using the **Random Forest Classifier**. By carefully examining demographic and financial attributes, we developed a robust model that achieved high accuracy and interpretability, making it suitable for deployment in real-world applications.

### Key Learnings
- **Feature Engineering**: Extracting meaningful information was critical to enhancing model performance.
- **Model Selection**: Highlighted the effectiveness of ensemble models, especially in handling non-linear data patterns.
- **ROC Curve and AUC Score**: These metrics were instrumental in assessing model performance and expanded my understanding of classification metrics.

### Further Recommendations and Improvements
- **Model Tuning**: Further hyperparameter tuning could potentially improve the model's accuracy.
- **Exploring More Complex Models**: Consider trying **Gradient Boosting** or **XGBoost** for additional accuracy gains.
- **Class Imbalance Handling**: Balancing the dataset could enhance the model's ability to classify approvals accurately.












