# Section B
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv("food_final.csv")

# Step 2: Encode the target variable 'FoodGroup'
label_encoder = LabelEncoder()
data['FoodGroup'] = label_encoder.fit_transform(data['FoodGroup'])

# Step 3: Define input variables
X = data.drop(columns=['FoodGroup', 'ID'])  # Drop 'FoodGroup' and 'ID' for input

# Step 4: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 6: Compute explained variance ratio
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(explained_variance_ratio >= 0.90) + 1

# Step 7: Print the top 5 Eigenvalues and Eigenvectors
eigenvalues = pca.explained_variance_[:5]
eigenvectors = pca.components_[:5]
print(f"Number of components capturing 90% variance: {n_components_90}")
print("Top 5 Eigenvalues:", eigenvalues)
print("Top 5 Eigenvectors:")
print(eigenvectors)

# Reduce dataset using the selected number of components
pca_90 = PCA(n_components=n_components_90)
X_reduced = pca_90.fit_transform(X_scaled)

# Step 8: Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_reduced)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 9: Use silhouette score to validate optimal clusters
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, labels)
    silhouette_scores.append(score)

optimal_clusters = np.argmax(silhouette_scores) + 2
print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")

# Step 10: Fit the final KMeans model with 4 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=42)
data['Cluster_4'] = kmeans_4.fit_predict(X_reduced)

# Step 11: Analyze clusters and sort by WCSS
cluster_inertia = []
for cluster in range(4):
    cluster_data = X_reduced[data['Cluster_4'] == cluster]
    inertia = np.sum((cluster_data - kmeans_4.cluster_centers_[cluster]) ** 2)
    cluster_inertia.append((cluster, inertia))

# Sort clusters by inertia
cluster_inertia.sort(key=lambda x: x[1])
print("Clusters ordered by WCSS (Inertia):")
for cluster, inertia in cluster_inertia:
    print(f"Cluster {cluster}: WCSS = {inertia}")

# Step 12: Perform hierarchical clustering and plot dendrogram
linkage_matrix = linkage(X_reduced[:100], method='ward')  # Use 'ward' linkage and top 100 samples
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Dendrogram for Hierarchical Clustering (Top 100 Samples)')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Section C
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the data
data = pd.read_csv("food_final.csv")

# Step 2: Encode the target variable 'FoodGroup'
label_encoder = LabelEncoder()
data['FoodGroup'] = label_encoder.fit_transform(data['FoodGroup'])

# Step 3: Define input and output variables
X = data.drop(columns=['FoodGroup', 'ID'])  # Drop 'FoodGroup' and 'ID' for input
y = data['FoodGroup']

# Step 4: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def apply_dimensionality_reduction(X_scaled, method='PCA', n_components=0.95):
    if method == 'PCA':
        pca = PCA(n_components=n_components)  # Retain 95% of variance
        X_reduced = pca.fit_transform(X_scaled)
    elif method == 'SVD':
        svd = TruncatedSVD(n_components=10, random_state=42)  # Reduce to 10 components
        X_reduced = svd.fit_transform(X_scaled)
    else:
        X_reduced = X_scaled  # No dimensionality reduction
    return X_reduced

def train_and_evaluate_model(X_reduced, y):
    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    # Step 6: Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Step 7: Make predictions
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return accuracy, conf_matrix, class_report

# Evaluate without dimensionality reduction
print("Evaluation without Dimensionality Reduction:")
accuracy, conf_matrix, class_report = train_and_evaluate_model(X_scaled, y)
print("Accuracy:", accuracy)

# Evaluate with PCA
print("\nEvaluation with PCA:")
X_pca = apply_dimensionality_reduction(X_scaled, method='PCA')
accuracy, conf_matrix, class_report = train_and_evaluate_model(X_pca, y)
print("Accuracy:", accuracy)

# Evaluate with SVD
print("\nEvaluation with SVD:")
X_svd = apply_dimensionality_reduction(X_scaled, method='SVD')
accuracy, conf_matrix, class_report = train_and_evaluate_model(X_svd, y)
print("Accuracy:", accuracy)

# Step 1: Load the dataset
data = pd.read_csv('amazon_ratings_Musical_Instruments.csv')

# Step 2: Calculate the popularity of each item
item_popularity = data.groupby('ItemID').agg(
    average_rating=('Rating', 'mean'),
    num_ratings=('Rating', 'count')
).reset_index()

# Step 3: Sort items by number of ratings (most popular first)
item_popularity = item_popularity.sort_values(by='num_ratings', ascending=False)

# Step 4: Recommend top 5 items based on popularity
top_5_items = item_popularity.head(5)

# Display the top 5 recommended items
print("Top 5 Recommended Items Based on Popularity:")
print(top_5_items[['ItemID', 'average_rating', 'num_ratings']])

# Step 1: Load the dataset
data = pd.read_csv('Recommendation_mini.csv')

# Step 2: Drop the timestamp column as per the requirement
data = data.drop(columns=['Timestamp'])

# Step 3: Prepare data for collaborative filtering using the Surprise library
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import numpy as np
reader = Reader(rating_scale=(data['Rating'].min(), data['Rating'].max()))
dataset = Dataset.load_from_df(data[['UserID', 'ItemID', 'Rating']], reader)

# Step 4: Use train_test_split to split the data
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Step 5: Train the SVD model (Singular Value Decomposition)
svd = SVD()
svd.fit(trainset)

# Step 6: Evaluate the model using RMSE on the test set
predictions = svd.test(testset)
rmse = np.sqrt(mean_squared_error([true_r for (_, _, true_r, _) in predictions], [est for (_, _, _, est) in predictions]))
print(f'RMSE: {rmse}')

# Step 7: Recommend top 5 items for a specific user
user_id = 1  # Specify the user ID for whom to recommend items
user_ratings = data[data['UserID'] == user_id]

# Get the list of items that the user hasn't rated yet
rated_items = user_ratings['ItemID'].tolist()
all_items = data['ItemID'].unique()
unrated_items = [item for item in all_items if item not in rated_items]

# Predict the ratings for the unrated items
predictions = [svd.predict(user_id, item) for item in unrated_items]

# Sort the predictions by predicted rating and get the top 5 items
top_5_items = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]

# Output the top 5 recommended items
print("Top 5 Recommended Items for User ID:", user_id)
for pred in top_5_items:
    print(f"ItemID: {pred.iid}, Predicted Rating: {pred.est}")


# ##############################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances, confusion_matrix
from yellowbrick.cluster import SilhouetteVisualizer
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Load dataset (hypothetical, replace with actual dataset)
X = pd.read_csv('hypothetical_dataset.csv')

# -----------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------

# Check for missing values
print("Missing Values:\n", X.isnull().sum())

# Summary statistics
print("\nSummary Statistics:\n", X.describe())

# Visualize data distribution for selected features
sns.histplot(X['feature_1'], kde=True)
plt.title("Distribution of Feature 1")
plt.show()

# ---------------------------
# Pre-processing (Standardize)
# ---------------------------

# Handle missing values (fill with median or remove rows)
X.fillna(X.median(), inplace=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------
# Principal Component Analysis (PCA)
# ----------------------

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Print top 5 Eigenvalues and Eigenvectors
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

print("Top 5 Eigenvalues:", eigenvalues[:5])
print("Top 5 Eigenvectors:\n", eigenvectors[:5])

# ----------------------
# K-Means Clustering with Principal Components
# ----------------------

# Reduce data to 90% explained variance
pca_90 = PCA(n_components=0.90)
X_pca_90 = pca_90.fit_transform(X_scaled)

# Determine optimal number of clusters for KMeans
silhouette_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    visualizer = SilhouetteVisualizer(kmeans, X=X_pca_90)
    visualizer.fit(X_pca_90)
    silhouette_scores.append(visualizer.silhouette_score_)

# Plot silhouette scores for each number of clusters
plt.plot(range_n_clusters, silhouette_scores)
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for K-Means Clustering")
plt.show()

# ----------------------
# Dendrograms and Cophenetic Correlation
# ----------------------

# Linkage methods to compare
linkage_methods = ['single', 'complete', 'average', 'centroid', 'ward']

# Plot dendrograms for each linkage method and calculate cophenetic correlation
for method in linkage_methods:
    plt.figure(figsize=(10, 7))
    Z = sch.linkage(X_pca_90, method=method)
    sch.dendrogram(Z)
    plt.title(f'Dendrogram ({method.capitalize()} Linkage)')
    plt.show()

    # Calculate cophenetic correlation coefficient
    cophenetic_corr = sch.cophenet(Z, pairwise_distances(X_pca_90))[0]
    print(f"Cophenetic Correlation ({method} linkage):", cophenetic_corr)

# ----------------------
# Clustering Without PCA (K-Means and Agglomerative)
# ----------------------

# K-Means clustering without PCA
kmeans_no_pca = KMeans(n_clusters=3, random_state=42)
kmeans_no_pca_labels = kmeans_no_pca.fit_predict(X_scaled)

# Agglomerative Clustering without PCA
agglo_no_pca = AgglomerativeClustering(n_clusters=3)
agglo_no_pca_labels = agglo_no_pca.fit_predict(X_scaled)

# Compare the cluster labels using confusion matrix
print("\nConfusion Matrix (K-Means vs Agglomerative):")
print(confusion_matrix(kmeans_no_pca_labels, agglo_no_pca_labels))

# Visualize the clustering results
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_no_pca_labels, cmap='viridis')
plt.title("K-Means Clustering without PCA")
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agglo_no_pca_labels, cmap='viridis')
plt.title("Agglomerative Clustering without PCA")
plt.show()

# Part C - Market Basket Analysis with Apriori and Collaborative Filtering

# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ------------ Market Basket Analysis with Apriori ------------

# Step 1: Load the dataset
market_basket_data = pd.read_csv('market_basket.csv')

# Step 2: Pre-processing
# Convert the data into a transaction format where each row represents a transaction and each column represents an item.
# For simplicity, we assume each item is a column, and each row is a transaction

# Let's assume each transaction has an ID and items are columns with 1 if present and 0 if absent.
# Example of pre-processing to create the format:
basket_data = market_basket_data.groupby('TransactionID')['Item'].apply(list).reset_index()
transactions = basket_data['Item'].apply(lambda x: {item: True for item in x}).tolist()

# Step 3: Apply Apriori algorithm
from mlxtend.preprocessing import TransactionEncoder

# Convert list format to 0,1 matrix for apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 4: Build the Apriori model with a minimum support of 10%
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

# Step 5: Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=2)

# Step 6: Print values of Lift greater than 2
print("Association Rules with Lift > 2:")
print(rules[rules['lift'] > 2][['antecedents', 'consequents', 'lift']])

# ------------ Collaborative Filtering using SVD ------------

# Step 1: Load the dataset
hotel_data = pd.read_csv('hotel_ratings.csv')

# Step 2: Pre-process the data
reader = Reader(rating_scale=(hotel_data['Rating'].min(), hotel_data['Rating'].max()))
dataset = Dataset.load_from_df(hotel_data[['UserID', 'HotelID', 'Rating']], reader)

# Step 3: Train-test split
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Step 4: Train the SVD model
svd = SVD()
svd.fit(trainset)

# Step 5: Evaluate the model using RMSE
predictions = svd.test(testset)
rmse = np.sqrt(mean_squared_error([true_r for (_, _, true_r, _) in predictions], [est for (_, _, _, est) in predictions]))
print(f'RMSE: {rmse}')

# Step 6: Recommend top hotel for a specific user (for example, user_id = 1)
user_id = 1  # Specify the user ID
user_ratings = hotel_data[hotel_data['UserID'] == user_id]

# Get list of hotels that the user hasn't rated yet
rated_hotels = user_ratings['HotelID'].tolist()
all_hotels = hotel_data['HotelID'].unique()
unrated_hotels = [hotel for hotel in all_hotels if hotel not in rated_hotels]

# Predict the ratings for unrated hotels
predictions = [svd.predict(user_id, hotel) for hotel in unrated_hotels]

# Sort predictions by estimated rating and get the top 5 hotels
top_5_hotels = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]

# Output the top 5 recommended hotels
print(f"Top 5 Recommended Hotels for User ID {user_id}:")
for pred in top_5_hotels:
    print(f"HotelID: {pred.iid}, Predicted Rating: {pred.est}")

