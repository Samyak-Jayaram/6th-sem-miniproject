import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay


file_path = 'correct_cleaned_china_data.csv'
data = pd.read_csv(file_path)


print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())


print("\nMissing Values:")
print(data.isnull().sum())


plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()


plt.figure(figsize=(10, 6))
sns.histplot(data['AC'], kde=True)
plt.title('Distribution of AC')
plt.savefig('ac_distribution.png')
plt.close()


features = data.columns.drop('AC')
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature, y='AC', data=data)
    plt.title(f'AC vs {feature}')
    plt.savefig(f'ac_vs_{feature}.png')
    plt.close()


plt.figure(figsize=(12, 6))
data.boxplot(figsize=(12, 6))
plt.title('Boxplots for All Features')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('boxplots.png')
plt.close()


X = data.drop(columns=['AC'])
y = data['AC']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


features_to_plot = feature_importance['feature'].head(3).tolist()  # Top 3 important features
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(rf, X, features_to_plot, ax=ax)
plt.tight_layout()
plt.savefig('partial_dependence_plots.png')
plt.close()


y_pred = rf.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('residual_plot.png')
plt.close()


plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot")
plt.savefig('qq_plot.png')
plt.close()

print("\nAnalysis complete. Please check the generated PNG files for visualizations.")