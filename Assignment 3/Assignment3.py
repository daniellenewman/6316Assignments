import pandas as pd
import numpy as np
from random import uniform

data = pd.read_csv("housing.csv")

# Drop rows containing NaN values (only total_bedrooms does)
data = data.dropna()

def KMeans(K, data, dist_measure):
    # Initialize K centroids randomly
    centroids = []
    for i in range(K):
        centroids.append([])
        for feature in data.columns:
            centroids[i].append(uniform(data[feature].min(), data[feature].max()))
    
    # Assign all data to centroids
    centroid_assignments = {}
    for index, row in data.iterrows():
        centroid_assignments[index] = assign_to_cluster(centroids, row, dist_measure)
    
    # Recalculate centroids and reassign data until there is no change
    new_centroid_assignments = {}
    while(True):
        if centroid_assignments = new_centroid_assignments:
            return centroids, centroid_assignments
        
    

def assign_to_cluster(centroids, sample, dist_measure):
    if dist_measure == "euclidean":        
        # Find which centroid the sample is the closest to
        # according to the Euclidean distance measure
        min_distance = -np.inf
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