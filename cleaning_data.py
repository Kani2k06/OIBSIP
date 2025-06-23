# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load Dataset
df = pd.read_csv('AB_NYC.csv')  # Make sure the file is in your working directory
print("Dataset Shape:", df.shape)
print("First 5 rows:\n", df.head())

# Step 3: Check Column Info and Missing Values
print("\nDataset Info:")
df.info()

print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Handle Missing Data
# Fill missing 'reviews_per_month' with 0 (no reviews), and drop rows with missing 'name' or 'host_name'
df['reviews_per_month'].fillna(0, inplace=True)
df.dropna(subset=['name', 'host_name'], inplace=True)

# Step 5: Remove Duplicates
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"\nRemoved {before - after} duplicate rows.")

# Step 6: Standardization (e.g., lowercase all neighborhood names)
df['neighbourhood'] = df['neighbourhood'].str.lower()
df['neighbourhood_group'] = df['neighbourhood_group'].str.lower()

# Step 7: Outlier Detection for Price
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['price'])
plt.title("Price Distribution with Outliers")
plt.show()

# Remove extreme price outliers (e.g., price > $1000)
df = df[df['price'] <= 1000]

# Step 8: Save Cleaned Dataset
df.to_csv("cleaned_airbnb_nyc.csv", index=False)
print("\n✅ Data cleaned and saved as 'cleaned_airbnb_nyc.csv'")
