import statistics

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import shap

df = pd.read_csv("CleanedMentalHealth.csv")

boolColumns = df.select_dtypes(include='bool').columns
df[boolColumns] = df[boolColumns].astype(int)

X = df.drop("treatment", axis=1)
y = df["treatment"]

Xsample = shap.sample(X, 1000, random_state=42)

rf = pickle.load(open("models/RandomForest.pkl", "rb"))
xg = pickle.load(open("models/XGBoost.pkl", "rb"))

def shapRF():
    explainerRf = shap.PermutationExplainer(rf.predict_proba, Xsample)
    shapRf = explainerRf(Xsample)

    shap_rf_values = shapRf.values[:, :, 1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_rf_values, Xsample, show=False)
    plt.title("Random Forest SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

def shapXG():
    explainerXG = shap.TreeExplainer(
        xg,
        feature_perturbation="tree_path_dependent",
        model_output="raw"
    )

    shapXG = explainerXG.shap_values(Xsample, approximate=True)

    if isinstance(shapXG, list):
        shap_xg_values = shapXG[1]
    else:
        shap_xg_values = shapXG

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_xg_values, Xsample, show=False)
    plt.title("XGBoost SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

shapXG()
