import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("CleanedMentalHealth.csv")

print(df["Days_Indoors"].unique())

features = ["treatment", "Coping_Struggles", "Growing_Stress", "Mood_Swings"]

combined = []

for i in features:
    temp = (
        df[df[i] == 1]
        .groupby("Gender")[i]
        .count()
        .div(df.groupby("Gender")[i].count())
        .mul(100)
        .reset_index(name="Percentage")
    )
    temp["Variable"] = i
    combined.append(temp)

dfCombined = pd.concat(combined, ignore_index=True)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    data=dfCombined,
    x="Gender",
    y="Percentage",
    hue="Variable",
    errorbar=None
)

for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x() + p.get_width()/2,
        y=height + 1,
        s=f'{height:.2f}%',
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title("Gender Comparison: Treatment & Emotional Struggles")
plt.ylabel("%")
plt.xlabel("Gender")
plt.legend(title="Category")
plt.tight_layout()
plt.ylim(0, 100)
plt.show()


dfStudents = df[df['Occupation'] == 'Student']

emotionalFeatures = ['Mood_Swings', 'Coping_Struggles', 'Growing_Stress']

dfGrouped = (
    dfStudents.groupby('Days_Indoors')[emotionalFeatures]
    .mean()
    .reset_index()
)

dfMelted = dfGrouped.melt(
    id_vars='Days_Indoors',
    value_vars=emotionalFeatures,
    var_name='Emotional_Factor',
    value_name='Proportion'
)

days = ['Go out Every day', '1-14 days', '15-30 days', '31-60 days', 'More than 2 months']
dfMelted['Days_Indoors'] = pd.Categorical(dfMelted['Days_Indoors'], categories=days, ordered=True)

plt.figure(figsize=(12,6))
ax = sns.barplot(
    data=dfMelted,
    x='Days_Indoors',
    y='Proportion',
    hue='Emotional_Factor'
)

for p in ax.patches:
    height = p.get_height()
    ax.text(
        x=p.get_x() + p.get_width() / 2,
    y=height + 0.02,
        s=f'{height*100:.1f}%',
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title("Students' Emotional Struggles by Days Spent Indoors")
plt.ylabel("Proportion of Students")
plt.xlabel("Days Indoors")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.legend(title="Emotional Factor")
plt.tight_layout()
plt.show()