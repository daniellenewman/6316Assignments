import pandas as pd
import numpy as np
from random import uniform

data = pd.read_csv("housing.csv")

# Drop rows containing NaN values (only total_bedrooms does)
data = data.dropna()

def KMeans(K, data, dist_measure = "euclidean"):
    # Initialize K centroids randomly
    centroids = pd.DataFrame(columns = data.columns, index = range(K))
    for i in range(K):
        for feature in data.columns:
            centroids.at[i, feature] = uniform(data[feature].min(), data[feature].max())
    
    # Assign data to clusters
    cluster_assignments = pd.DataFrame(columns = ["Cluster"], index = data.index)
    for index, row in data.iterrows():
        cluster_assignments.at[index, "Cluster"] = assign_to_cluster(centroids, row, dist_measure)
    
    # Recompute centroids
    averages = pd.DataFrame(columns = data.columns, index = range(K))
    averages.fillna(0)
    for index, row in cluster_assignments.iterrows():
        for feature in data.columns:
            averages.at[row["Cluster"], feature] += data.at[index, feature]
    
    for index, row in averages.iterrows():
        for feature in data.columns:
            
    
    return centroids, centroid_assignments

#def KMeans(K, data, dist_measure = "euclidean"):
#    # Initialize K centroids randomly
#    centroids = []
#    for i in range(K):
#        centroids.append([])
#        for feature in data.columns:
#            centroids[i].append(uniform(data[feature].min(), data[feature].max()))
#    
#    # Initialize centroid_assignments data structure
#    centroid_assignments = {}
#    for i in range(K):
#        centroid_assignments[i] = []
#    
#    # Assign all data to centroids
#    for index, row in data.iterrows():
#        centroid_assignments[assign_to_cluster(centroids, row, dist_measure)].append(index)
#
#    while True:
#        # Recompute centroids
#        for centroid in centroid_assignments:
#            # Reset averages of all features to 0
#            averages = []
#            for i in range(len(data.columns)):
#                averages.append(0)
#            
#            # Iterate over samples in this cluster,
#            # adding their features to average
#            for sample in centroid_assignments[centroid]:
#                for i in range(len(data.columns)):
#                    averages[i] += data.at[sample, data.columns[i]]
#            
#            # Divide all sums by number of samples in cluster to get average
#            for i in range(len(averages)):
#                if(len(centroid_assignments[centroid]) != 0):
#                    averages[i] /= len(centroid_assignments[centroid])
#            
#            # Re-assign centroid values
#            for i in range(len(centroids[centroid])):
#                centroids[centroid][i] = averages[i]
#        
#        # Reassign data to clusters
#        new_centroid_assignments = {}
#        for i in range(K):
#            new_centroid_assignments[i] = []
#        
#        for index, row in data.iterrows():
#            new_centroid_assignments[assign_to_cluster(centroids, row, dist_measure)].append(index)
#        
#        # If clusters have not changed, we're done
#        if new_centroid_assignments == centroid_assignments:
#            return centroid_assignments
#        else:
#            centroid_assignments = new_centroid_assignments
        
    
#TODO: Fix this to assume centroids is a dataframe
def assign_to_cluster(centroids, sample, dist_measure):
    if dist_measure == "euclidean":
        # Find which centroid the sample is the closest to
        # according to the Euclidean distance measure
        min_distance = np.inf
        best_centroid = -1
        for i in range(len(centroids)):
            distance = euclidean_distance(centroids[i], sample)
            
            if distance < min_distance:
                min_distance = distance
                best_centroid = i
        
        return best_centroid
            
    elif dist_measure == "spearson":
        pass
    else:
        print("Error in assign_to_cluser(): Unrecognized distance measure")


def euclidean_distance(vector1, vector2):
    if(len(vector1) != len(vector2)):
        print("Error in euclidean_distance(): Vectors must be of equal length")
        return -1

    return np.sqrt(np.sum(np.subtract(vector1, vector2) ** 2))

def spearson_distance(vector1, vector2):
    pass