import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Obective 1 --> data cleaning
data1 = pd.read_csv("death_suicide.csv")
print(data1)

print("Information of the dataset: ",data1.info())
print("\nDescription of the dataset: ",data1.describe())
print("\nSum of all the null shells in the dataset: ",data1.isnull().sum())
print("\nTop 5 rows of the dataset: ",data1.head(5))
print("\nsum of all the duplicate values: ",data1.duplicated().sum())
me = data1['ESTIMATE'].mean()
data1.fillna({'ESTIMATE' : me}, inplace = True)
print("\nsum of null values in estimate columm: ",data1['ESTIMATE'].isnull().sum())
print("\ndescription of the dataset after removal of null values from estimate column: \n",data1.describe())

#objective 2 --> Suicide Rate Trend Over Time (Overall)
overall_data = data1[(data1['STUB_LABEL'] == 'All persons') & (data1['AGE'] == 'All ages')]
plt.figure(figsize=(12, 6))
sns.lineplot(data=overall_data, x = 'YEAR', y = 'ESTIMATE', marker='*', color='red')
plt.title('Suicide Rate Trend Over Time (All Persons, All Ages)')
plt.xlabel('Year')
plt.ylabel('Suicide Rate per 100,000 Population')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#objective 3 -->  Suicide Rates by Gender Over Time
gender_data = data1[
    (data1['STUB_LABEL'].str.contains('Male|Female')) &
    (data1['AGE'] == 'All ages')
].copy()
gender_data.loc[:, 'Gender'] = gender_data['STUB_LABEL'].str.extract(r'(Male|Female)')
plt.figure(figsize=(12, 6))
sns.lineplot(data=gender_data, x='YEAR', y='ESTIMATE', hue='Gender', marker='o')
plt.title('Suicide Rates by Gender Over Time (All Ages)')
plt.xlabel('Year')
plt.ylabel('Suicide Rate per 100,000 Population')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend(title='Gender')
plt.show()

#Objective 4 --> Suicide Rates by Race and Gender (for a single year like 2018)
race_gender_data = data1[
    (data1['STUB_LABEL'].str.contains('Male|Female')) &
    (data1['AGE'] == 'All ages') &
    (data1['YEAR'] == 2018)
].copy()

race_gender_data_sorted = race_gender_data.sort_values(by='ESTIMATE', ascending=True)

plt.figure(figsize=(12, 8))
sns.barplot(data=race_gender_data_sorted, y='STUB_LABEL', hue = 'STUB_LABEL',x='ESTIMATE', palette='viridis',dodge = False,legend =False )
plt.title('Suicide Rates by Race and Gender (2018)')
plt.xlabel('Suicide Rate per 100,000')
plt.ylabel('Race and Gender Group')
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()

#Objective 5--> Gender-wise Distribution of Suicide Rates (Pie Chart)
gender_data = data1[data1['STUB_LABEL'].isin(['Male', 'Female'])]
gender_counts = gender_data.groupby('STUB_LABEL')['ESTIMATE'].sum()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title("Suicide Rate Distribution by Gender")
plt.axis('equal')
plt.show()

#Objective 6 --> Yearly Suicide
gender_data = data1[data1['STUB_LABEL'].isin(['Male', 'Female'])]
heatmap_data = gender_data.groupby(['YEAR', 'STUB_LABEL'])['ESTIMATE'].mean().unstack()
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=0.5)
plt.title("Yearly Suicide Estimates by Gender")
plt.xlabel("Gender")
plt.ylabel("Year")
plt.tight_layout()
plt.show()
yearly_avg = data1.groupby('YEAR')['ESTIMATE'].mean().reset_index()
heatmap_data = [yearly_avg['ESTIMATE'].values]
plt.figure(figsize=(12, 2))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".1f", xticklabels=yearly_avg['YEAR'], yticklabels=["Average"])
plt.title("Year-wise Suicide Estimates (Heatmap)")
plt.xlabel("Year")
plt.tight_layout()
plt.show()

#objective 7 -->Distribution of Suicide Estimates (KDE Plot)
plt.figure(figsize=(8, 5))
sns.kdeplot(data1['ESTIMATE'], fill=True, color='green')
plt.title("Distribution of Suicide Estimates (KDE Plot)")
plt.xlabel("Estimate")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

#objective 8 -->Correlation Heatmap (n x n)
numeric_cols = data1.select_dtypes(include='number')
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#objective 9 --> Histogram of Suicide Estimates
plt.hist(data1['ESTIMATE'], bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Suicide Estimates")
plt.xlabel("Estimate")
plt.ylabel("Frequency")
plt.show()

#objective 10 --> Suicide Estimates by Age Group (Scatterplot)
age_data = data1[data1['AGE'] != 'All ages']
avg_age = age_data.groupby('AGE')['ESTIMATE'].mean().reset_index()
plt.scatter(avg_age['AGE'], avg_age['ESTIMATE'], color='orange')
plt.xticks(rotation=45)
plt.title("Average Suicide Estimate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Estimate")
plt.tight_layout()
plt.show()

#objective 11 --> Line Plot of Yearly Suicide Averages
yearly_avg = data1.groupby('YEAR')['ESTIMATE'].mean().reset_index()
plt.plot(yearly_avg['YEAR'], yearly_avg['ESTIMATE'], marker='*', color='teal')
plt.title("Yearly Average Suicide Estimate")
plt.xlabel("Year")
plt.ylabel("Estimate")
plt.grid(True)
plt.show()

data1.to_csv("Cleaned_Death_Sucide.csv")



