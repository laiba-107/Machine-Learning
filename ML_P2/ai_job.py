import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("ai_job_replacement.csv")
df
print(df.shape)
print(df.info())
print(df.describe())
df[df.isnull().any(axis=1)]
#corelation Heatmap

# compute correlation matrix
corr = df.corr(numeric_only=True)

# plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
#drop unnessassary columns
df = df.drop(columns=['job_id'], axis=1)
num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(exclude='number').columns

print('Numberic:', len(num_cols))
print('Categorical:', len(cat_cols))

print(cat_cols)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# 4 categorical columns
cat_cols = ['job_role', 'industry', 'country', 'automation_risk_category']

# Encode each column in-place
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])  # fill NaNs with mode
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print(df[cat_cols].head())
df
#numeric to median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill all categorical columns with mode

# Here we need to user median for categorical columns as all are now converted to numeric values
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].median())


print(df[cat_cols].head())
# #print(df.isnull())
# null_rows = df[df.isnull().any(axis=1)]
# print(null_rows)
target = "ai_disruption_intensity"
#seperate X and Y
X = df.drop(target, axis=1)
y = df[target]
#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

#KNN Regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)

print("\nKNN")

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_knn)))
print("MAE:", mean_absolute_error(y_test, pred_knn))
print("R2:", r2_score(y_test, pred_knn))

#DECISION Tree
#KNN Regressor
dt = DecisionTreeRegressor(max_depth=30, random_state=42)
dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)


print("\nDecition Tree")
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_dt)))
print("MAE:", mean_absolute_error(y_test, pred_dt))
print("R2:", r2_score(y_test, pred_dt))

#Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("\nRandom Forest")

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))
print("MAE:", mean_absolute_error(y_test, pred_rf))
print("R2:", r2_score(y_test, pred_rf))

# AdaBoost
ada = AdaBoostRegressor(random_state=42)
ada.fit(X_train, y_train)

pred_ada = ada.predict(X_test)

print("\nAdaBoost")

print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_ada)))
print("MAE:", mean_absolute_error(y_test, pred_ada))
print("R2:", r2_score(y_test, pred_ada))

results = {
    "Linear": rmse_lr,
    "KNN": np.sqrt(mean_squared_error(y_test, pred_knn)),
    "Tree": np.sqrt(mean_squared_error(y_test, pred_dt)),
    "RF": np.sqrt(mean_squared_error(y_test, pred_rf)),
    "Ada": np.sqrt(mean_squared_error(y_test, pred_ada))
}

print(results)