import csv
import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader
from scipy.cluster.hierarchy import dendrogram, linkage

def load_data(filename):
    dict_reader = DictReader(open(filename, 'r'))
    data = [dict(o_dict) for o_dict in list(dict_reader)]
    return data  

def calc_features(row):
    arr = np.array([row['Attack'], row['Sp. Atk'], row['Speed'], \
                row['Defense'], row['Sp. Def'], row['HP']], dtype=int)
    return arr

def hac(features):
  n = len(features)
  clusters = {i: [i] for i in range(n)} 
  Z = [[0]*4 for i in range(n-1)]
  clus_count = 0

  # Calculate all distances and store in an array
  dist_arr = [[0]*n for i in range(0, n)]
  for i in range(0, n):
    for j in range(0, n):
      dist_arr[i][j] = np.linalg.norm(features[i] - features[j])

  for iterations in range(n-1):
    #Finding which clusters to merge
    clus1 = 0
    clus2 = 0
    dist_min = np.inf

    keys = list(clusters.keys())
    for i in range(len(keys)):
      for j in range(i+1, len(keys)):
        c1 = keys[i]
        c2 = keys[j]
        dist_max = -np.inf

        for x in clusters[c1]:
          for y in clusters[c2]:
            if dist_arr[x][y] > dist_max:
              dist_max = dist_arr[x][y]

        if(dist_max == dist_min):
          print('Found', dist_max, dist_min)

        if(dist_max < dist_min):
          dist_min = dist_max
          clus1 = c1
          clus2 = c2

    #Merging the two cluster into a new cluster, and removing the two single clusters
    clusters[n + clus_count] = clusters[clus1] + clusters[clus2]
    len_merged = len(clusters[n + clus_count])
    clus_count += 1
    clusters.pop(clus1, None)
    clusters.pop(clus2, None)

    #Filling the necessary elements in Z array
    Z[clus_count - 1][0] = clus1
    Z[clus_count - 1][1] = clus2
    Z[clus_count - 1][2] = dist_min
    Z[clus_count - 1][3] = len_merged

  return np.array(Z)

def imshow_hac(Z):
  plt.figure()
  dn = dendrogram(Z)
  plt.show()