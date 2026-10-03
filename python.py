import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_samples = 100

data = {
    'Student_ID': np.arange(1001, 1001 + n_samples),
    'Study_Hours': np.random.uniform(5, 30, size=n_samples),
    'Attendance_Rate': np.random.uniform(60, 100, size=n_samples),
    'Exam_Score': np.random.randint(40, 100, size=n_samples),
    'Course_Category': np.random.choice(['Data Science', 'Web Dev', 'UX Design'], size=n_samples)
}

df = pd.DataFrame(data)

df.loc[df['Study_Hours'] > 28, 'Study_Hours'] = np.nan

print("--- 1. FIRST 5 ROWS ---")
print(df.head())

print("\n--- 2. DATASET SUMMARY & CLEANING ---")
print(df.info())
print("\nMissing values per column before fix:\n", df.isnull().sum())

study_hours_median = df['Study_Hours'].median()
df['Study_Hours'].fillna(study_hours_median, inplace=True)

print("Missing values after cleaning:", df['Study_Hours'].isnull().sum())

print("\n--- 3. EXPLORATORY DATA ANALYSIS ---")

print(df.describe())

avg_scores = df.groupby('Course_Category')['Exam_Score'].mean().reset_index()
print("\nAverage Exam Score by Course Category:\n", avg_scores)

df['Performance_Status'] = np.where(df['Exam_Score'] >= 75, 'Passing', 'Needs Improvement')

# Setting up standard visual context style
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Capstone Data Operations & Analysis Dashboard', fontsize=16, fontweight='bold')

sns.histplot(data=df, x='Exam_Score', kde=True, ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Distribution of Exam Scores')

sns.scatterplot(data=df, x='Study_Hours', y='Exam_Score', hue='Performance_Status', palette='Set1', ax=axes[0,1])
axes[0,1].set_title('Study Hours vs. Exam Score')

sns.boxplot(data=df, x='Course_Category', y='Attendance_Rate', ax=axes[1,0], palette='Pastel1')
axes[1,0].set_title('Attendance Rate Distribution by Course')

numeric_df = df[['Study_Hours', 'Attendance_Rate', 'Exam_Score']]
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1,1])
axes[1,1].set_title('Numerical Correlation Heatmap')

plt.tight_layout()
plt.show()
