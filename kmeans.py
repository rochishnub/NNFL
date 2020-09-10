import numpy as np
import pandas as pd 
import math

# euclidean distance
def d_euclid(row1, row2):
    diff = np.array(row1) - np.array(row2)
    return np.linalg.norm(diff)

class k_means():
    def __init__(self, k, dataset, iter):
        self.k = k
        self.dataset = dataset.copy()
        self.n_data, self._variables = dataset.shape
        self.iter = iter

        self.clusters = [[] for i in range(self.k)]
        self.centroids = []
    
    def clustering(self):
        # pick centroids
        cent_ind = np.random.choice(self.n_data, self.k, replace = False)
        self.centroids.append(self.dataset[i] for i in cent_ind)

        # clustering algorithm
        for i in range(iter):
            # clustering datapoints
            self.clusters = self.init_cluster(self.centroids)

            # optimize centroid
            old_centroids = self.centroids
            self.centroids = self.opt_centroid(self.clusters)

            # check error function
            if self.check_error(old_centroids, self.centroids):
                break
        
        # final clusters
        return self.final_clusters(self.clusters)

    # clustering datapoints
    def init_cluster(self, centroids):
        clusters = [[] for i in range(self.k)]
        for i, sample in enumerate(self.dataset):
            cent_ind = self.nearest_centroid(centroids, sample)
            clusters[cent_ind].append(i)
        return clusters

    # nearest centroid
    def nearest_centroid(self, centroids, sample):
        distances = [d_euclid(sample, center) for center in centroids]
        return np.argmin(distances)
    
    # optimizing centroids
    def opt_centroid(self, clusters):
        new_centroids = np.zeros((self.k, self._variables))
        for i, cluster in enumerate(self.clusters):
            new_centroids[i] = np.mean(self.dataset[cluster])
            return new_centroids
    
    # error function
    def check_error(self, old_centroids, centroids):
        error = [d_euclid(old_centroids[i], centroids[i]) for i in self.k]
        if sum(error) == 0:
            return True
    
    # final clusters
    def final_clusters(self, clusters):
        final = np.empty(self.n_data)
        for n_cluster, cluster in enumerate(clusters):
            for ind in cluster:
                final[ind] = n_cluster
        return final
    
    # file


    
    



    

# create dataset from csv file
# def create_dataset(filename):
    # dataset = pd.read_csv(filename, header = 0, index_col = None)
    #



    


