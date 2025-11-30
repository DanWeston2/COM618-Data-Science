import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("Mental Health Dataset.csv")

df.drop_duplicates(inplace=True)

df.drop(columns=["Country", "mental_health_interview", "Timestamp"], inplace=True)

yesNoMap = {"Yes": 1, "No": 0}
yesNoMaybeMap = {"Yes": 1, "No": 0, "Maybe": 0.5, "Not sure": 0.5}

df["family_history"] = df["family_history"].map(yesNoMap)
df["treatment"] = df["treatment"].map(yesNoMap)
df["Coping_Struggles"] = df["Coping_Struggles"].map(yesNoMap)
df["Growing_Stress"] = df["Growing_Stress"].map(yesNoMaybeMap)
df["Changes_Habits"] = df["Changes_Habits"].map(yesNoMaybeMap)
df["Mental_Health_History"] = df["Mental_Health_History"].map(yesNoMaybeMap)
df["Mood_Swings"] = df["Mood_Swings"].map({"High": 1, "Low": 0, "Medium": 0.5})
df["Work_Interest"] = df["Work_Interest"].map(yesNoMaybeMap)
df["Social_Weakness"] = df["Social_Weakness"].map(yesNoMaybeMap)
df["care_options"] = df["care_options"].map(yesNoMaybeMap)

df.dropna(inplace=True)

zScores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(zScores < 3).all(axis=1)]

dfMale = df[df["Gender"] == "Male"].sample(n=40000, random_state=42)
dfFemale = df[df["Gender"] == "Female"].sample(n=40000, random_state=42)
dfSample = pd.concat([dfMale, dfFemale]).sample(frac=1, random_state=42).reset_index(drop=True)

dfSample.to_csv("CleanedMentalHealth.csv", index=False, header=True)