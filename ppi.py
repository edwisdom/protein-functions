from __future__ import division
import networkx as nx
import numpy as np 
import sys, os, re, codecs, time, collections
from numpy.linalg import inv, LinAlgError
from scipy.spatial.distance import pdist, squareform
from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn import metrics
np.random.seed(1)


# Initialize filenames
NETWORK_DATA = "ppidata.txt"
MIPS1 = "mips1.txt" # First-level labels
MIPS2 = "mips2.txt" # Second-level labels
MIPS3 = "mips3.txt" # Third-level labels


# Reads in edges line-by-line from a text file and returns a graph 
def read_network(filename):
	G = nx.Graph()
	with open(filename, 'r') as f:
		for line in f:
			words = line.split()
			G.add_edge(words[0], words[1])
	return G

# Reads in a text file line-by-line and returns a dict with proteins as keys,
# and a list of labels as values.
def read_labels(filename):
	labels = {}
	with open(filename, 'r') as f:
		for line in f:
			words = line.split()
			labels[words[0]] = words[1:] 
	return labels

def split_data(graph):
	print(graph)

def calculator(adjacency, nRw):
	# This function can replace the calculator() function in the calcDSD.py file
    # This is the original function in the calcDSD.py file (for comparison purposes)

    """
    adjacency - adjacency matrix represented as a numpy array
                assumes graph is fully connected.

    nRW - the length of random walks used to calculate DSD
          if nRW = -1, then calculate DSD at convergence

    returns DSD matrix represented as a numpy array
    """
    adjacency = np.asmatrix(adjacency)
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    p = adjacency / degree
    if nRw >= 0:
        c = np.eye(n)
        for i in range(nRw):
            c = np.dot(c, p) + np.eye(n)
        return squareform(pdist(c,metric='cityblock'))
    else:
        pi = degree / degree.sum()
        # The inverse of this singular matrix can't be computed, so we 
        # "regularize" along the diagonals. The matrix is unstable, so
        # this operation should likely be avoided or modified.
        try:
        	return squareform(pdist(inv(np.eye(n) - p - pi.T),metric='cityblock'))
        except LinAlgError:
        	return squareform(pdist(inv(np.eye(n) - p - pi.T + (np.eye(n) * 10e-9)),metric='cityblock'))

# Prints out basic attributes of the network
def show_basic_attributes(G):
	print ("Network holds " + str(len(G.nodes)) + " proteins.")
	print ("There are " + str(len(G.edges)) + " edges.")
	total_labels = 0
	for i in mips1.values():
		total_labels += len(i)
	print ("We have labels for " + str(len(mips1.keys())) + " proteins.")
	print ("There are " + str(total_labels) + " total labels.") 


# Conducts a small exploratory data analysis on the network
def show_eda(mips):
	# Shows the frequency distribution of the NUMBER of labels
	labels = mips.values()
	lengths = list(map(len, labels))
	plt.figure(1)
	plt.hist(lengths, bins=np.arange(1, 12), align='left', rwidth=0.5, histtype='bar')
	plt.title('Frequency Distribution of the Number of Labels')
	plt.xlabel('Number of Labels')
	plt.ylabel('Number of Proteins')

	# Shows the frequency distribution of EACH particular label
	plt.figure(2)
	flattened = [item for sublist in labels for item in sublist]
	plt.hist(flattened, bins=range(19), align='left', rwidth=0.5, histtype='barstacked')
	plt.title('Frequency Distribution of the Labels')
	plt.xlabel('Labels')
	plt.ylabel('Number of Proteins')

	# Visualizes the correlations between labels as a heatmap
	unique_labels = list(set(flattened))
	idx_len = len(unique_labels)
	matrix_i = np.zeros((idx_len,idx_len), dtype=int)
	corr_matrix = pd.DataFrame(data=matrix_i, index=unique_labels, columns=unique_labels, dtype=np.int)
	for label_set in labels:
		set_length = len(label_set)
		for i in range(set_length):
			l1 = label_set[i]
			corr_matrix[l1][l1] += 1
			for j in range(i+1, set_length):
				l2 = label_set[j]
				corr_matrix[l1][l2] += 1
				corr_matrix[l2][l1] += 1
	diagonal = np.diag(corr_matrix)
	normalized_corr = corr_matrix / diagonal
	plt.figure(3, figsize=(14,10))
	sns.heatmap(normalized_corr, annot=True)

	# Plot degree distribution
	# NEEDS WORK, FIX THE X-AXIS 
	degrees = ([tup[-1] for tup in G.degree()])
	plt.figure(4, figsize=(14, 10))
	sns.countplot(degrees)
	plt.title("Degree Histogram")
	plt.ylabel("Count")
	plt.xlabel("Degree")
	plt.show()


# IN PROGRESS: Clustering the DSD Matrix
# A = nx.adjacency_matrix(G).toarray()
# dsd_A = calculator(A, 3)
# gt_dict = nx.get_node_attributes(G, 'labels')
# sc = SpectralClustering(1000`, affinity='rbf', assign_labels='kmeans', n_jobs=-1)



if __name__ == '__main__':
	# Read in network and labels using the above functions
	G = read_network(NETWORK_DATA)
	#split_data(G)
	mips1 = read_labels(MIPS1)
	nx.set_node_attributes(G, mips1, 'labels')
	#split_data(G)

	show_basic_attributes(G)
	show_eda(mips1)