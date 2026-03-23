import pandas as pd
import numpy as np

df = pd.read_csv("college_student_placement_syn.csv")
df
print(df.shape)
print(df.info())
print(df.describe())
df.isnull()
#drop unnessassary columns
df = df.drop(columns=['College_ID'], axis=1)
num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(exclude='number').columns

print('Numberic:', len(num_cols))
print('Categorical:', len(cat_cols))


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
#print(df.isnull())
null_rows = df[df.isnull().any(axis=1)]
print(null_rows)
# Fill numeric columns with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
df['Internship_Experience'] = df['Internship_Experience'].fillna(df['Internship_Experience'].mode()[0])
df['Placement'] = df['Placement'].fillna(df['Placement'].mode()[0])
# Convert 'Yes'/'No' into 1/0
yes_no_cols = ['Internship_Experience', 'Placement']
for col in yes_no_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

print(df[yes_no_cols].head())
df
# define the target variable or output class
target = "Placement"
#seperate X and Y
X = df.drop(target, axis=1)
y = df[target]
#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=.3, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
# # Create models dictionary
# models = {
#     "LinearRegression": LinearRegression(),
#     "KNN": KNeighborsRegressor(n_neighbors=5),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "AdaBoost": AdaBoostClassifier(random_state=42)
# }
#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)

print("LinearRegression")
print("RMSE:", rmse_lr)
print("MAE:", mae_lr)
print("R2:", r2_lr)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)
pred_knn_proba = knn.predict_proba(X_test)[:, 1]  # Probability for class 1

print("\n=== KNN Classifier ===")
print("Accuracy:", accuracy_score(y_test, pred_knn))
print("Precision:", precision_score(y_test, pred_knn))
print("Recall:", recall_score(y_test, pred_knn))
print("F1-Score:", f1_score(y_test, pred_knn))

# Specificity = TN / (TN + FP)
cm = confusion_matrix(y_test, pred_knn)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

print("ROC-AUC:", roc_auc_score(y_test, pred_knn_proba))

# Confusion Matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('KNN Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, pred_knn_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='KNN (AUC = {:.3f})'.format(roc_auc_score(y_test, pred_knn_proba)))
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
#DECISION Tree
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)
pred_dt_proba = dt.predict_proba(X_test)[:, 1]  # Probability for class 1


print("\n=== Decision Tree Classifier ===")
print("Accuracy:", accuracy_score(y_test, pred_dt))
print("Precision:", precision_score(y_test, pred_dt))
print("Recall:", recall_score(y_test, pred_dt))
print("F1-Score:", f1_score(y_test, pred_dt))

# Specificity = TN / (TN + FP)
cm = confusion_matrix(y_test, pred_dt)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

print("ROC-AUC:", roc_auc_score(y_test, pred_dt_proba))

# Confusion Matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('dt Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, pred_dt_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='dt (AUC = {:.3f})'.format(roc_auc_score(y_test, pred_dt_proba)))
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#Random Forest
rf = RandomForestClassifier(n_estimators=40, criterion='gini', max_depth=20,  random_state=5)
rf.fit(X_train, y_train)

pred_rf = knn.predict(X_test)
pred_rf_proba = knn.predict_proba(X_test)[:, 1]  # Probability for class 1

print("\n=== Random Forest Classifier ===")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("Precision:", precision_score(y_test, pred_rf))
print("Recall:", recall_score(y_test, pred_rf))
print("F1-Score:", f1_score(y_test, pred_rf))

# Specificity = TN / (TN + FP)
cm = confusion_matrix(y_test, pred_rf)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

print("ROC-AUC:", roc_auc_score(y_test, pred_rf_proba))

# Confusion Matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, pred_rf_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='KNN (AUC = {:.3f})'.format(roc_auc_score(y_test, pred_rf_proba)))
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.01, random_state=42)
ada.fit(X_train, y_train)

pred_ada = ada.predict(X_test)
pred_ada_proba = ada.predict_proba(X_test)[:, 1]  # Probability for class 1

print("\n=== AdaBoost Classifier ===")
print("Accuracy:", accuracy_score(y_test, pred_ada))
print("Precision:", precision_score(y_test, pred_ada))
print("Recall:", recall_score(y_test, pred_ada))
print("F1-Score:", f1_score(y_test, pred_ada))

# Specificity = TN / (TN + FP)
cm = confusion_matrix(y_test, pred_ada)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

print("ROC-AUC:", roc_auc_score(y_test, pred_ada_proba))

# Confusion Matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('AdaBoost Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, pred_ada_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ada (AUC = {:.3f})'.format(roc_auc_score(y_test, pred_ada_proba)))
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)
pred_nb_proba = nb.predict_proba(X_test)[:, 1]  # Probability for class 1

print("\n=== AdaBoost Classifier ===")
print("Accuracy:", accuracy_score(y_test, pred_nb))
print("Precision:", precision_score(y_test, pred_nb))
print("Recall:", recall_score(y_test, pred_nb))
print("F1-Score:", f1_score(y_test, pred_nb))

# Specificity = TN / (TN + FP)
cm = confusion_matrix(y_test, pred_ada)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

print("ROC-AUC:", roc_auc_score(y_test, pred_nb_proba))

# Confusion Matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Naive Bayes Classifier Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, pred_nb_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label='ada (AUC = {:.3f})'.format(roc_auc_score(y_test, pred_ada_proba)))
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
