import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('DataSet/Mall_Customers.csv')
# mengganti nama kolom
df = df.rename(columns={
    'Gender' : 'gender',
    'Age' : 'age',
    'Annual Income (k$)' : 'annual_income',
    'Spending Score (1-100)' : 'speding_score'
})
# ubah data kategorik menjadi data numerik
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)
# menghilangkan kolom CustomerID
new_data = df.drop(['CustomerID', 'gender'], axis=1)
# membuat list yang berisi inertia
cluster = []
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(new_data)
    cluster.append(km.inertia_)
# membuat plot inertia
fig, ax = plt.subplots(figsize=(8,4))
sns.lineplot(x=list(range(1, 11)), y = cluster, ax=ax)
ax.set_title('Cari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# membuat objek KMeans
km5 = KMeans(n_clusters=5).fit(new_data)
 
# menambahkan kolom label pada dataset
new_data['Labels'] = km5.labels_
 
# membuat plot KMeans dengan 5 klaster
plt.figure(figsize=(8,4))
sns.scatterplot(x=new_data['annual_income'], y=new_data['speding_score'], hue=new_data['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('KMeans dengan 5 Cluster')
plt.show()
# print(df)
# print(cluster)   