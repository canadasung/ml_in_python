# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

hostname = 'your_hostPath'
username = 'your_userName'
password = 'your_password'
database = 'your_database'


# Pull data from PostgresDatabase
print ("Using psycopg2…")
import psycopg2
myConnection = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )
cur = myConnection.cursor()

# Prepare the Query
sql_query = """you full sql queries here;"""

# Write to Pandas DataFrame
new_df = pd.read_sql(sql_query, myConnection)

# Close the connection
myConnection.close()

# Read data from drive
new_df = pd.read_csv("C:/Users/user_name/Documents/your_file.csv")

# DataFrame Information
new_df.info()
new_df = new_df.sort_values(by = ['total_purchases'], ascending = False) #sort by specific column
new_df['purchase_lapsed'] = new_df['purchase_lapsed'].fillna(-1)  #fill na with -1
new_df.dropna(inplace = True) #drop na and overwrite the dataframe
dataset = new_df

# Importing the dataset
#dataset= pd.read_csv("file.csv")
dataset = new_df[pd.notnull(new_df['tenure'])] #drop rows where null in tenure column
dataset = dataset[dataset.tenure >= 0] #drop negative value in tenure column
dataset = dataset.fillna(0) #fill na with 0 for total_purchases and purchase_tenure columns
dataset.reset_index(drop = True, inplace = True)

dataset.info()
X = dataset.drop(columns =['columnName1', 'columnName2','columnName3', 'columnName4', 'columnName5'])
X = X.drop(columns =['lapsed','sessions_count','purchase_lapsed','purchase_tenure']) #drop unnecessary columns
X = X.drop(columns =['tenure'])
X.info()
#X.fillna(0)
#X = dataset.iloc[:, [1, 10]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Using the elbow method to find the optimal number of clusters
# Change range(1,12) to setup your own range
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 12):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 12), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
# Change n_clusters to the optimal value from previous elbow method
kmeans = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10000, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10000, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10000, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10000, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Write back the segment to dataframe
df = pd.DataFrame(y_kmeans)   #get the group value from y_kmeans
df.info()
verticalStack = pd.concat([dataset, df], axis = 1)  #concatenate to original dataset
verticalStack.rename(columns = {0:'seg'}, inplace = True)  #rename the column
print(verticalStack.columns)   
verticalStack.info()   # Double check the dataset
verticalStack.to_csv('seg4-2.csv')  #write to drive with given name

