import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

train_data = pd.read_excel('train.xlsx')

X = train_data.drop('target',axis=1)
#scaling the features
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
train_data['Cluster'] = kmeans.fit_predict(X_scaled)

feature_names = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18']

centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)


# function to identify the cluster where the data point belongs to
def predict_cluster(data_point):
    # transform the data_point
    data_point_scaled = scaler.transform([data_point])

    # Predict the cluster
    cluster_label = kmeans.predict(data_point_scaled)[0]

    # Retrieve the cluster center
    cluster_center = centers_df.loc[cluster_label]

    # Find the most common target value(s) in this cluster
    cluster_members = train_data[kmeans.labels_ == cluster_label]
    most_common_target = cluster_members['target'].value_counts().idxmax()

    explanation = f"The data point belongs to cluster {cluster_label} because it is closest to the cluster center with the following characteristics:\n"
    for feature, value in zip(feature_names, data_point_scaled[0]):
        center_value = cluster_center[feature]
        explanation += f"Feature {feature}: {value:.2f} (Data Point) vs {center_value:.2f} (Cluster Center)\n"

    return cluster_label, most_common_target, explanation