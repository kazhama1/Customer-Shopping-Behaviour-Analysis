
# ============================================================
# PROJECT: Customer Shopping Behaviour Analysis and
#          Purchase Pattern Discovery Using Data Mining
# ============================================================

# STEP 1: IMPORT REQUIRED LIBRARIES

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

# Create a folder for project results
os.makedirs("project_results", exist_ok=True)

print("=" * 65)
print(" CUSTOMER SHOPPING BEHAVIOUR ANALYSIS ")
print("=" * 65)


# STEP 2: LOAD THE DATASET

print("\n[1] Loading Dataset...")

FILE_PATH = "Online Retail.xlsx"

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        "Dataset not found! Place Online Retail.xlsx "
        "in the same folder as this Python file."
    )

df = pd.read_excel(FILE_PATH)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# STEP 3: DATA PREPROCESSING

print("\n[2] Preprocessing Dataset...")

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove records with missing customer IDs
df.dropna(subset=["CustomerID", "Description"], inplace=True)

# Remove cancelled transactions
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]

# Remove invalid quantities and prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Convert invoice dates into datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Calculate total purchase amount
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Clean product descriptions
df["Description"] = (
    df["Description"].astype(str).str.strip().str.upper()
)

# Convert customer IDs into integers
df["CustomerID"] = df["CustomerID"].astype(int)

print("Data Preprocessing Completed!")
print("Cleaned Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv(
    "project_results/cleaned_dataset.csv",
    index=False
)


# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)

print("\n[3] Performing Exploratory Data Analysis...")

# 4.1 Most Popular Products

top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(11, 6))
sns.barplot(
    x=top_products.values,
    y=top_products.index
)

plt.title("Top 10 Most Purchased Products")
plt.xlabel("Quantity Purchased")
plt.ylabel("Product")

plt.tight_layout()
plt.savefig("project_results/top_products.png")
plt.show()


# 4.2 Customer Spending Distribution

customer_spending = (
    df.groupby("CustomerID")["TotalAmount"].sum()
)

plt.figure(figsize=(10, 5))
sns.histplot(
    customer_spending,
    bins=30,
    kde=True
)

plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("project_results/customer_spending.png")
plt.show()


# 4.3 Most Popular Product Categories

# The dataset contains product descriptions rather
# than predefined product categories.

plt.figure(figsize=(10, 5))

top_items = (
    df.groupby("Description")["Quantity"]
    .sum()
    .nlargest(10)
)

plt.pie(
    top_items.values,
    labels=top_items.index,
    autopct="%1.1f%%",
    startangle=140
)

plt.title("Distribution of Top 10 Purchased Products")
plt.tight_layout()
plt.savefig("project_results/product_distribution.png")
plt.show()


# 4.4 Monthly Sales Trend

monthly_sales = (
    df.set_index("InvoiceDate")
    .resample("ME")["TotalAmount"]
    .sum()
)

plt.figure(figsize=(12, 5))
monthly_sales.plot(marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("project_results/monthly_sales.png")
plt.show()


# STEP 5: CUSTOMER SEGMENTATION USING K-MEANS

print("\n[4] Applying K-Means Clustering...")

# Create customer-level RFM features

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

customer_data = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate",
             lambda x: (reference_date - x.max()).days),

    Frequency=("InvoiceNo", "nunique"),

    Monetary=("TotalAmount", "sum")
)

print("\nCustomer Behaviour Data:")
print(customer_data.head())

# Log transformation reduces the effect of extreme values
customer_features = np.log1p(customer_data)

# Standardize the features
scaler = StandardScaler()

scaled_data = scaler.fit_transform(customer_features)

# Elbow Method to examine suitable cluster numbers
inertia = []

for k in range(2, 8):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_data)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 8),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.tight_layout()
plt.savefig("project_results/elbow_method.png")
plt.show()


# Apply K-Means with four customer clusters

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = kmeans.fit_predict(scaled_data)

# Calculate silhouette score
score = silhouette_score(scaled_data, customer_data["Cluster"])

print("\nK-Means Clustering Completed!")
print("Silhouette Score:", round(score, 3))


# Display cluster information

cluster_summary = customer_data.groupby("Cluster").agg(
    Customer_Count=("Recency", "count"),
    Average_Recency=("Recency", "mean"),
    Average_Frequency=("Frequency", "mean"),
    Average_Spending=("Monetary", "mean")
)

print("\nCustomer Cluster Summary:")
print(cluster_summary.round(2))

customer_data.to_csv(
    "project_results/customer_segments.csv"
)

cluster_summary.to_csv(
    "project_results/cluster_summary.csv"
)


# Visualize customer clusters

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=customer_data,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="Set2",
    s=70
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spending")

plt.tight_layout()
plt.savefig("project_results/customer_clusters.png")
plt.show()


# STEP 6: MARKET BASKET ANALYSIS USING APRIORI

print("\n[5] Applying Apriori Algorithm...")

# Group products by transaction ID

transactions = (
    df.groupby("InvoiceNo")["Description"]
    .apply(lambda x: list(set(x)))
)

# Retain the 40 most frequently purchased products
# to make the analysis computationally manageable.

product_frequency = (
    df.groupby("Description")["InvoiceNo"]
    .nunique()
    .sort_values(ascending=False)
)

top_40_products = set(product_frequency.head(40).index)

filtered_transactions = [
    [item for item in transaction if item in top_40_products]
    for transaction in transactions
]

# Remove transactions with no selected products
filtered_transactions = [
    transaction
    for transaction in filtered_transactions
    if len(transaction) > 0
]

print("Total Transactions:", len(filtered_transactions))

# Convert transactions into one-hot encoded format

encoder = TransactionEncoder()

encoded_array = encoder.fit(
    filtered_transactions
).transform(filtered_transactions, sparse=True)

basket_data = pd.DataFrame.sparse.from_spmatrix(
    encoded_array,
    columns=encoder.columns_
)

print("\nTransaction Matrix Created!")
print(basket_data.head())


# Apply Apriori Algorithm

frequent_itemsets = apriori(
    basket_data,
    min_support=0.01,
    use_colnames=True,
    low_memory=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head(10))


# Generate Association Rules

if not frequent_itemsets.empty:

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.20
    )

    if not rules.empty:

        rules = rules.sort_values(
            by="lift",
            ascending=False
        )

        print("\nTop Association Rules:")

        print(
            rules[
                ["antecedents", "consequents",
                 "support", "confidence", "lift"]
            ].head(15)
        )

        # Save association rules
        rules.to_csv(
            "project_results/association_rules.csv",
            index=False
        )

        # Save frequent itemsets
        frequent_itemsets.to_csv(
            "project_results/frequent_itemsets.csv",
            index=False
        )

        # Visualize association rules

        plt.figure(figsize=(9, 6))

        sns.scatterplot(
            data=rules,
            x="support",
            y="confidence",
            size="lift",
            hue="lift",
            sizes=(40, 300),
            palette="viridis"
        )

        plt.title("Apriori Association Rules")
        plt.xlabel("Support")
        plt.ylabel("Confidence")

        plt.tight_layout()
        plt.savefig("project_results/apriori_rules.png")
        plt.show()

    else:
        print("No association rules found.")
        print("Try reducing the minimum confidence threshold.")

else:
    print("No frequent itemsets found.")
    print("Try reducing the minimum support threshold.")


# STEP 7: FINAL PROJECT SUMMARY

print("\n" + "=" * 65)
print(" PROJECT ANALYSIS COMPLETED ")
print("=" * 65)

print("\nPROJECT SUMMARY")

print("Total Transactions:", df["InvoiceNo"].nunique())

print("Total Customers:", df["CustomerID"].nunique())

print("Total Unique Products:", df["Description"].nunique())

print(
    "Total Sales Amount:",
    round(df["TotalAmount"].sum(), 2)
)

print("Number of Customer Clusters:", 4)

print("K-Means Silhouette Score:", round(score, 3))

print(
    "Number of Frequent Itemsets:",
    len(frequent_itemsets)
)

if "rules" in locals():
    print("Number of Association Rules:", len(rules))

print("\nResults have been saved in the project_results folder.")

print("\nThank you for using the Customer Shopping Analysis System!")
