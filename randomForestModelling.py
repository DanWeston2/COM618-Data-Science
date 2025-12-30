import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def SaveModel(path, model):
    modelDirectory = 'models'
    os.makedirs(modelDirectory, exist_ok=True)

    with open(os.path.join(modelDirectory, path + '.pkl'), 'wb') as f:
        pickle.dump(model, f)

df = pd.read_csv("CleanedMentalHealth.csv")

X = df.drop("treatment", axis=1)
y = df["treatment"]

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

paramGrid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [10, 20, 40, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ['sqrt', 'log2'],
    "bootstrap": [True, False],
}

search = RandomizedSearchCV(
    rf,
    param_distributions=paramGrid,
    n_iter=20,
    cv=3,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

search.fit(Xtrain, yTrain)
bestModel = search.best_estimator_

yPred = bestModel.predict(Xtest)

SaveModel("RandomForest", bestModel)

print(f"Accuracy: {accuracy_score(yTest, yPred)}")
print(classification_report(yTest, yPred))
print(confusion_matrix(yTest, yPred))

importance = bestModel.feature_importances_
features = X.columns

dfImportance = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

dfImportance.head(10)

cm = confusion_matrix(yTest, yPred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10,6))
plt.barh(dfImportance['feature'], dfImportance['importance'])
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
