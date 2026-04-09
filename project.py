# we pasted this in from our group google colab

from google.colab import files
fitness = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#Q1
df["Intensity"] = df["Calories_Burned"] / df["Session_Duration (hours)"]

df["Age_Group"] = pd.cut(df["Age"], bins=[15, 25, 35, 45, 55, 65],
                          labels=["16-25", "26-35", "36-45", "46-55", "56-65"])

sns.boxplot(data=df, x="Age_Group", y="Calories_Burned", hue="Age_Group", palette="Set2")
plt.title("Distribution of Calories Burned by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Calories Burned")
plt.show()


#Q2
corr = df[['Experience_Level', 'Workout_Frequency (days/week)', 'Age']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


fig, ax = plt.subplots()
with sns.axes_style(style=None):
    sns.violinplot(x="Experience_Level", y="Workout_Frequency (days/week)", hue="Gender",
                   data=df, split=True, inner="quartile",
                   palette=["lightblue", "lightpink"], ax=ax)

ax.set_xlabel("Experience Level")
ax.set_ylabel("Workout Frequency (days/week)")
ax.set_title("Experience Level vs Workout Frequency by Gender")
plt.show()



#Q3
kwargs = dict(histtype='step', alpha=0.8, bins=20)
level1 = df[df["Experience_Level"] == 1]["Fat_Percentage"]
level2 = df[df["Experience_Level"] == 2]["Fat_Percentage"]
level3 = df[df["Experience_Level"] == 3]["Fat_Percentage"]

plt.figure()
plt.hist(level1, label="Level 1 (Beginner)", **kwargs)
plt.hist(level2, label="Level 2 (Intermediate)", **kwargs)
plt.hist(level3, label="Level 3 (Advanced)", **kwargs)

sns.kdeplot(level1, fill=False)
sns.kdeplot(level2, fill=False)
sns.kdeplot(level3, fill=False)

plt.title("Body Fat Distribution by Experience Level")
plt.xlabel("Body Fat Percentage")
plt.ylabel("Frequency")
plt.legend()
plt.show()


#Q4
sns.scatterplot(
    data=df,
    x='Avg_BPM',
    y='Calories_Burned',
    hue='Workout_Type'
)
plt.title("BPM vs Calories Burned by Workout Type")
plt.show()


#Q5
water_median = np.median(df['Water_Intake (liters)'])

low_water  = df['Fat_Percentage'][df['Water_Intake (liters)'] <  water_median]
high_water = df['Fat_Percentage'][df['Water_Intake (liters)'] >= water_median]

sns.kdeplot(low_water,  label='Low Water Intake',  fill=True)
sns.kdeplot(high_water, label='High Water Intake', fill=True)

plt.xlabel('Fat Percentage')
plt.title('Fat Percentage by Water Intake Level')
plt.legend()
plt.show()


#Q6
x = df['Session_Duration (hours)'].values
y = df['Calories_Burned'].values
 
plt.scatter(x, y)
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('Session Duration vs Calories Burned')
plt.show()
 
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)
 
xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict(xfit[:, np.newaxis])
 
plt.scatter(x, y)
plt.plot(xfit, yfit, color='red')
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('Linear Regression: Calories Burned ~ Session Duration')
plt.show()
 
print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)
