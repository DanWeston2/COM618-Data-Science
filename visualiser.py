import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Mental Health Dataset.csv")

'''
columns = [
    'Growing_Stress',
    'Mood_Swings',
    'Coping_Struggles',
    'Work_Interest',
    'Social_Weakness'
]
binaryDict = {'Yes': 1, 'No': 0}
moodDict = {'Low': 0, 'Medium': 1, 'High': 2}
dfNumeric = df[columns].copy()
dfNumeric['Growing_Stress'] = dfNumeric['Growing_Stress'].map(binaryDict)
dfNumeric['Work_Interest'] = dfNumeric['Work_Interest'].map(binaryDict)
dfNumeric['Mood_Swings'] = dfNumeric['Mood_Swings'].map(moodDict)
dfNumeric['Coping_Struggles'] = dfNumeric['Coping_Struggles'].map(binaryDict)
dfNumeric['Social_Weakness'] = dfNumeric['Social_Weakness'].map(binaryDict)
corr = dfNumeric.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Stress and Mood Variables")
plt.show()
'''

'''
stress_percent = (
    df.groupby('Days_Indoors')['Growing_Stress']
    .value_counts(normalize=True)
    .mul(100)
    .rename('percentage')
    .reset_index()
)

sns.barplot(
    data=stress_percent,
    x='Days_Indoors',
    y='percentage',
    hue='Growing_Stress'
)
plt.title('Growing Stress by Days Indoors')
plt.ylabel('Percentage (%)')
plt.xlabel('Days Indoors')
plt.xticks(rotation=45)
plt.legend(title='Growing Stress')
plt.tight_layout()
plt.show()
'''

'''
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.show()
'''

'''
sns.countplot(data=df, x="Occupation")
plt.title("Occupation Distribution")
plt.show()
'''

'''
sns.countplot(data=df, x="family_history", hue="treatment")
plt.title("Treatment Rates by Family History")
plt.show()
'''

'''
treatment_percent = (
    df.groupby('Country')['treatment']
    .value_counts(normalize=True)
    .mul(100)
    .rename('percentage')
    .reset_index()
)
treatment_percent = treatment_percent[treatment_percent['treatment'] == 'Yes']
sns.barplot(
    data=treatment_percent,
    x='Country',
    y='percentage',
    color='skyblue'
)
plt.title('Percentage of Respondents Receiving Treatment by Country')
plt.ylabel('Treatment Rate (%)')
plt.xlabel('Country')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
'''

'''
mood_percent = (
    df.groupby('Work_Interest')['Mood_Swings']
    .value_counts(normalize=True)
    .mul(100)
    .rename('percentage')
    .reset_index()
)

sns.barplot(
    data=mood_percent,
    x='Work_Interest',
    y='percentage',
    hue='Mood_Swings'
)
plt.title('Mood Swings by Work Interest')
plt.ylabel('Percentage (%)')
plt.xlabel('Work Interest (Yes/No)')
plt.legend(title='Mood Swings')
plt.tight_layout()
plt.show()
'''

'''
Heatmap of correlation between stress and mood variables
'''