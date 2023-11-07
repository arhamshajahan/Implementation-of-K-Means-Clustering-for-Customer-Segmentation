# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

# AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Choose the number of clusters (K). 
2. Initialize cluster centroids. 
3. Assign data points to clusters. 
4. Update cluster centroids.
5. Repeat steps 3 and 4. 
6. Evaluate the clustering results. 
7. Select the best clustering solution.

# Program:
```py
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Arham s
RegisterNumber: 212222110005
```
```

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])
y_pred

data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")

```
# Output:
### data.head():
![image](https://github.com/Kowsalyasathya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118671457/447cdff2-1b7a-4da3-8508-425ac74898b4)
### data.info():
![image](https://github.com/Kowsalyasathya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118671457/6b6467d2-b450-4814-9928-05e99e181c4d)
### Null Values:
![image](https://github.com/Kowsalyasathya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118671457/e486a6e8-a4cc-4249-b75e-544ed64f6e00)
### Elbow Graph:
![image](https://github.com/Kowsalyasathya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118671457/933173c7-a93f-4635-834f-31de37fcd2dc)
### K-Means Cluster Formation:
![image](https://github.com/Kowsalyasathya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118671457/c862801a-8074-453a-9873-5767d094d6a8)

### Predicted Value:
![image](https://github.com/Kowsalyasathya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118671457/5551bf48-637d-447c-b3df-ec95e1781a9f)
### Final Graph:
![image](https://github.com/Kowsalyasathya/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118671457/2c6005b6-6174-4816-93f1-6af9070f8088)

# Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
