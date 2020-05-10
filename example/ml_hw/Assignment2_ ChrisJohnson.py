"""
Chris Johnson
Assigment 2
CS 3120-001
3/1/2020

Source -
teacher resource- 5_ExLogis_SKlearn.pdf
teacher demo- 2_PandasTest.py
"""


#   *******************************************************
#   Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for ROC

from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection


#   *******************************************************
#   1, 2 -
#   load dataset

column_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'labelvalue']
pima = pd.read_csv('pima-indians-diabetes-database.csv', header=None, names=column_names)
feature_columns = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
print(pima.head())
#   3 -

X = pima[feature_columns]  # Features
y = pima.labelvalue        # Target Values


#   *******************************************************
#   4 -
#   train test split

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.4, train_size = 0.6, random_state= 0)


#   *******************************************************
#   5 -
#   logistic regression

log_reg = linear_model.LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)


#   *******************************************************
#   6 -
#   Confusion Matrix simple

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("\n____________Confusion Matrix_____________")
print(confusion_matrix)


#   *******************************************************
#   visualize

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)


#   heatmap

sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


#   *******************************************************
#   Classification report

print("\n____________Classification Report_____________")
print(metrics.classification_report(y_test, y_pred))


#   *******************************************************
#   Precision, recall and F1

print("\nf1-score:",metrics.f1_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("\n")


#   *******************************************************
#   ROC Curve (Receiver Operating Characteristica)

y_pred_proba = log_reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

print("\n\nReceiver Operating Characteristic Score -", auc)

print("\ndone")
