from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data and use first two features
X = load_iris().data[:, :2]

# Fit K-means and predict clusters
y_pred = KMeans(n_clusters=3, random_state=42).fit_predict(X)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('K-means Clustering')
plt.show()

