# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sns.set(style="whitegrid")
df = pd.read_csv("Mall_Customers.csv")  
print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())
df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Spending'}, inplace=True)
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()
sns.scatterplot(x='Income', y='Spending', data=df, hue='Gender')
plt.title('Income vs Spending Score')
plt.show()
X = df[['Income', 'Spending']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
df['Cluster'] = y_kmean
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income', y='Spending', hue='Cluster', data=df, palette='tab10', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print("Cluster Summary:\n", cluster_summary)
print("\nBusiness Recommendations:")
print("- High Income & High Spending: Target with premium offers.")
print("- Low Income & High Spending: Promote value-based deals.")
print("- High Income & Low Spending: Encourage spending via loyalty rewards.")
print("- Low Income & Low Spending: Less priority segment.")
