import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Optimized_Dataset.csv')
X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 2].values

#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=.4, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifer=DecisionTreeClassifier()
classifer.fit(X,y)

y_pred= classifer.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# Making the Confusion Matrix
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm,annot=True)
plt.savefig('h.png')
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# Visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(classifer.fit(X_train,y_train), ax=ax)
plt.show()

import pickle 
# Save the trained model as a pickle string. 
with open ('DT_Classifier','wb') as f:
      pickle.dump(classifer,f)    