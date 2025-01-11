#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Section B


# In[10]:


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

# Find number of components capturing 90% variance
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

# X_reduced and data with cluster labels can be used for further business inferences.


# In[ ]:


# Section C


# In[11]:


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
#print("\nConfusion Matrix:\n", conf_matrix)
#print("\nClassification Report:\n", class_report)

# Evaluate with PCA
print("\nEvaluation with PCA:")
X_pca = apply_dimensionality_reduction(X_scaled, method='PCA')
accuracy, conf_matrix, class_report = train_and_evaluate_model(X_pca, y)
print("Accuracy:", accuracy)
#print("\nConfusion Matrix:\n", conf_matrix)
#print("\nClassification Report:\n", class_report)

# Evaluate with SVD
print("\nEvaluation with SVD:")
X_svd = apply_dimensionality_reduction(X_scaled, method='SVD')
accuracy, conf_matrix, class_report = train_and_evaluate_model(X_svd, y)
print("Accuracy:", accuracy)
#print("\nConfusion Matrix:\n", conf_matrix)
#print("\nClassification Report:\n", class_report)


# In[12]:


import pandas as pd

# Step 1: Load the dataset
data = pd.read_csv('amazon_ratings_Musical_Instruments.csv')

# Step 2: Calculate the popularity of each item
# Here, we'll calculate the average rating and the number of ratings for each item
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


# In[18]:


import pandas as pd
from sklearn.metrics import mean_squared_error
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv('Recommendation_mini.csv')

# Step 2: Drop the timestamp column as per the requirement
data = data.drop(columns=['Timestamp'])

# Step 3: Prepare data for collaborative filtering using the Surprise library
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


# In[20]:


get_ipython().system('wget')


# In[ ]:




The **cophenetic correlation coefficient** is a measure used in hierarchical clustering to assess how faithfully a dendrogram represents the pairwise distances between the original data points. It provides an indication of the quality of the hierarchical clustering.

### Key Points:
1. **Definition**:
   - The cophenetic distance between two data points is the height of the dendrogram at which the two points are first merged into a single cluster.
   - The **cophenetic correlation coefficient** compares the cophenetic distances with the original pairwise distances in the data.

3. **Purpose**:
   - Evaluates how well the hierarchical clustering captures the structure of the data.
   - A high cophenetic correlation (close to 1) indicates that the clustering results are a good representation of the original pairwise distances.

4. **Use Cases**:
   - To **validate the quality** of hierarchical clustering.
   - To **compare different linkage methods** (e.g., single, complete, average linkage) and choose the one that best preserves the data’s structure.
   - As a **benchmark** for clustering models when working with hierarchical methods.

5. **Interpretation**:
   - **\( r_c \approx 1\)**: The dendrogram accurately represents the original distances.
   - **\( r_c \approx 0\)**: The dendrogram poorly represents the original distances.
   - A threshold depends on the domain, but typically, a value > 0.75 is considered good.
```

### Summary:
- The cophenetic correlation coefficient is a valuable tool for understanding the quality of hierarchical clustering.
- It helps to decide whether the dendrogram is a reliable representation of the data structure.
- It's particularly useful for fine-tuning clustering methods and validating results.

Would you like assistance in computing the cophenetic correlation for your dataset or comparing linkage methods?


### Summary: Linkage Methods in Hierarchical Clustering

1. **Single Linkage**: Uses the shortest distance between two clusters. Detects irregular clusters but can lead to elongated ones due to the "chaining effect."

2. **Complete Linkage**: Uses the longest distance between two clusters. Produces compact clusters but is sensitive to outliers.

3. **Average Linkage (UPGMA)**: Uses the average of all pairwise distances. Balances single and complete linkage and is less sensitive to outliers.

4. **Weighted Average Linkage (WPGMA)**: Similar to average linkage but treats clusters equally regardless of size.

5. **Centroid Linkage**: Uses the distance between cluster centroids. Sensitive to centroid changes, may cause dendrogram inversions.

6. **Ward’s Method**: Minimizes within-cluster variance by reducing the increase in total squared error. Effective for compact, spherical clusters.

7. **Median Linkage (WPGMC)**: Based on the median of pairwise distances, similar to centroid linkage but less interpretable.

8. **Custom/Advanced Methods**: Includes flexible linkage for balancing single and complete linkage and maximum likelihood methods using probabilistic models.


### **Key Applications of SVD (Brief Overview)**

1. **Dimensionality Reduction**: Identifies principal components to reduce features in high-dimensional datasets, aiding in preprocessing and visualization.

2. **Data Compression**: Compresses images, audio, and videos by approximating data with the largest singular values.

3. **Recommender Systems**: Decomposes user-item matrices to discover latent factors for personalized recommendations.

4. **Noise Reduction**: Filters noise by reconstructing data with top singular values, used in signal and image denoising.

5. **Latent Semantic Analysis (LSA)**: Extracts semantic relationships from text data for applications like search engines and topic modeling.

6. **Pseudo-Inverse & Linear Systems**: Solves ill-conditioned or singular linear systems using the Moore-Penrose pseudo-inverse.

7. **Facial Recognition**: Identifies eigenfaces for biometric authentication and surveillance.

8. **Quantum Computing**: Models quantum states and operations, aiding in simulation and optimization.

9. **Signal & Image Processing**: Decom
    poses signals and images for analysis or reconstruction, useful in medical imaging and audio processing.

10. **Control Systems**: Analyzes system stability and rank in robotics and feedback design.

---

### **Strengths**:
- Handles high-dimensional, sparse, and noisy data effectively.
- Widely applicable in fields like machine learning, engineering, and natural sciences.

### **Market Basket Analysis (Brief Overview)**

Market Basket Analysis (MBA) identifies relationships between items frequently purchased together in transactional data to uncover patterns and inform business strategies.

---

### **Key Concepts**:
1. **Transaction**: A record of items purchased together (e.g., `{milk, bread, eggs}`).
2. **Itemset**: A group of items (e.g., `{milk, bread}` is a 2-itemset).
3. **Association Rule**: A relationship between items (e.g., "If bread → Then butter").
4. **Metrics**:
   - **Support**: Frequency of an itemset in transactions.
   - **Confidence**: Likelihood of a rule being true.
   - **Lift**: Strength of an association compared to random chance.

---

### **Techniques**:
1. **Apriori Algorithm**:
   - Iteratively finds frequent itemsets using the Apriori property (all subsets of a frequent itemset are frequent).
2. **FP-Growth Algorithm**:
   - Efficiently builds a compact FP-Tree to find frequent itemsets without candidate generation.
3. **Association Rule Mining**:
   - Generates rules with metrics like confidence and lift.

---

### **Applications**:
1. **Product Recommendations**: Suggest complementary items (e.g., batteries with electronics).
2. **Store Layout Optimization**: Place frequently bought-together items nearby.
3. **Promotions**: Bundle or discount frequently co-purchased products.
4. **Inventory Management**: Stock popular item combinations efficiently.
5. **Fraud Detection**: Spot unusual purchase patterns.
6. **E-commerce Personalization**: Enhance user experience with tailored recommendations.

---

**Conclusion**: MBA leverages frequent itemsets and association rules to improve customer experience, optimize operations, and boost revenue.

