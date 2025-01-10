# ### Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, RandomForestClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import warnings

# Suppress warnings and set display options
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.6f}'.format
plt.rcParams['figure.figsize'] = [15, 8]

# ### Load and Inspect Data
df = pd.read_csv('Admission_predict.csv')
print(df.head(), "\n")
print(f"Shape: {df.shape}")

# ### Data Preprocessing
# Drop irrelevant column
df.drop('Serial No.', axis=1, inplace=True)

# Correct data type for categorical variable
df['Research'] = df['Research'].astype('object')
print(df.dtypes)

# Check missing values
missing = df.isnull().sum()
print("Missing Values:\n", missing)

# ### Exploratory Data Analysis
# Visualize distributions
df.drop('Chance of Admit', axis=1).hist()
plt.tight_layout()
plt.show()

# Research count plot
sns.countplot(df['Research'])
plt.title('Count Plot - Research')
plt.show()

# ### Prepare Data for Modeling
# Split target and features
y = df['Chance of Admit']
X = df.drop('Chance of Admit', axis=1)

# Separate numeric and categorical features
X_num = X.select_dtypes(include=[np.number])
X_cat = X.select_dtypes(include=['object'])

# One-hot encode categorical variables
X_cat_dummies = pd.get_dummies(X_cat, drop_first=True)

# Combine numeric and dummy variables
X = pd.concat([X_num, X_cat_dummies], axis=1)

# ### Outlier Detection and Removal
def remove_outliers(df, lower_percentile=0.01, upper_percentile=0.99):
    """
    Removes outliers based on specified percentiles for numerical features.
    
    Parameters:
    df (DataFrame): Input DataFrame.
    lower_percentile (float): Lower percentile threshold (e.g., 0.01 for 1st percentile).
    upper_percentile (float): Upper percentile threshold (e.g., 0.99 for 99th percentile).
    
    Returns:
    DataFrame: DataFrame without outliers.
    """
    # Compute lower and upper bounds for each numerical column
    bounds = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)
        bounds[col] = (lower_bound, upper_bound)
    
    # Filter rows within the bounds for all numeric columns
    for col, (lower, upper) in bounds.items():
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    return df

# Apply outlier removal
print(f"Original data shape: {X.shape}")
X_cleaned = remove_outliers(X)
print(f"Data shape after outlier removal: {X_cleaned.shape}")
print(X_cleaned.head())

# ## 2.7 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, random_state=10, test_size=0.2)
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

# ### Model Evaluation Functions

# Generalized function to calculate the metrics for the test set
def get_test_report(model):    
    test_pred = model.predict(X_test)
    return classification_report(y_test, test_pred)

# Plot the confusion matrix
def plot_confusion_matrix(model):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = pd.DataFrame(data=cm, columns=['Predicted:0', 'Predicted:1'], index=['Actual:0', 'Actual:1'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=ListedColormap(['lightskyblue']), cbar=False, 
                linewidths=0.1, annot_kws={'size': 25})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

# Plot the ROC curve
def plot_roc(model):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC curve for Admission Prediction Classifier', fontsize=15)
    plt.xlabel('False positive rate (1-Specificity)', fontsize=15)
    plt.ylabel('True positive rate (Sensitivity)', fontsize=15)
    plt.text(x=0.82, y=0.3, s=('AUC Score:', round(roc_auc_score(y_test, y_pred_prob), 4)))
    plt.grid(True)

# ### Model Training and Evaluation

# **AdaBoost Classifier**
ada_model = AdaBoostClassifier(n_estimators=40, random_state=10)
ada_model.fit(X_train, y_train)
plot_confusion_matrix(ada_model)
test_report = get_test_report(ada_model)
print(test_report)
plot_roc(ada_model)

# **Gradient Boosting Classifier**
gboost_model = GradientBoostingClassifier(n_estimators=150, max_depth=10, random_state=10)
gboost_model.fit(X_train, y_train)
plot_confusion_matrix(gboost_model)
test_report = get_test_report(gboost_model)
print(test_report)
plot_roc(gboost_model)

# **XGBoost Classifier**
xgb_model = XGBClassifier(max_depth=10, gamma=1)
xgb_model.fit(X_train, y_train)
plot_confusion_matrix(xgb_model)
test_report = get_test_report(xgb_model)
print(test_report)
plot_roc(xgb_model)

# ### Hyperparameter Tuning for XGBoost
tuning_parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                     'max_depth': range(3, 10),
                     'gamma': [0, 1, 2, 3, 4]}
xgb_grid = GridSearchCV(estimator=XGBClassifier(), param_grid=tuning_parameters, cv=3, scoring='roc_auc')
xgb_grid.fit(X_train, y_train)
print('Best parameters for XGBoost classifier: ', xgb_grid.best_params_)

# Build model using the tuned hyperparameters
xgb_grid_model = XGBClassifier(learning_rate=xgb_grid.best_params_.get('learning_rate'),
                               max_depth=xgb_grid.best_params_.get('max_depth'),
                               gamma=xgb_grid.best_params_.get('gamma'))
xgb_model = xgb_grid_model.fit(X_train, y_train)
print('Classification Report for test set:\n', get_test_report(xgb_model))
plot_roc(xgb_model)

# **Feature Importance for XGBoost**
important_features = pd.DataFrame({'Features': X_train.columns, 'Importance': xgb_model.feature_importances_})
important_features = important_features.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Features', data=important_features)
plt.title('Feature Importance', fontsize=15)
plt.xlabel('Importance', fontsize=15)
plt.ylabel('Features', fontsize=15)
plt.show()
