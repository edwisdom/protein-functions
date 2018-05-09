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
	subgraph_l = nx.connected_component_subgraphs(G)
	max_G = list(subgraph_l)[0]
	for g in list(subgraph_l)[1:]:
		max_G = g if len(g.nodes) > len(max_G.nodes) else max_G
	return max_G

# Reads in a text file line-by-line and returns a dict with proteins as keys,
# and a list of labels as values.
def read_labels(filename):
	labels = {}
	with open(filename, 'r') as f:
		for line in f:
			words = line.split()
			labels[words[0]] = words[1:] 
	unique_labels = list(set([item for sublist in labels.values() for item in sublist]))
	return labels, unique_labels

def split_data(mips, split):
	train = {}
	test = {}
	for line in mips:
		if np.random.rand() > split:
			test[line] = mips[line]
		else:
			train[line] = mips[line]
		# try:
		# 	print(node, graph.node[node]['labels'])
		# 	if np.random.rand() > .7:
		# 		graph.node[node]['purpose'] = "validation"
		# 	else:
		# 		graph.node[node]['purpose'] = "train"
		# except:
		# 	print(node, " does not have any labels")
		# 	graph.node[node]['purpose'] = "test"
		# print("\tNode is for purpose:", graph.node[node]['purpose'])
	return (train, test)

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
	for i in mips.values():
		total_labels += len(i)
	print ("We have labels for " + str(len(mips.keys())) + " proteins.")
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

def getEvalMetrics():
	d = {}
	d['jaccard_score'] = compute_jaccard_score
	return d

def compute_jaccard_score(y_true, y_pred, normalize=True, sample_weight=None):
	avg_accuracy = 0
	for y in y_true:
		labels = y_true[y]
		try:
			predictions = y_pred[y]
		except:
			predictions = []
		lab_set = set(labels)
		pred_set = set(predictions)
		jac_sim = len(lab_set.intersection(pred_set)) / len(lab_set.union(pred_set))
		avg_accuracy += jac_sim
		#avg_accuracy += metrics.jaccard_similarity_score(labels, predictions, normalize, sample_weight)
	return avg_accuracy/len(y_true)

def mipsToLabels(mips, unique_labels):
	i = 0
	label_mat = np.zeros(len(mips)*len(unique_labels)).reshape(len(mips), len(unique_labels))
	key_index_map = {}
	for key in mips:
		key_index_map[key] = i
		orig_labs = mips[key]
		label_mat[i] = [int(label in orig_labs) for label in unique_labels]
		i+=1
	return label_mat, key_index_map

if __name__ == '__main__':
	# Read in network and labels using the above functions
	G = read_network(NETWORK_DATA)
	mips, unique_labels = read_labels(MIPS1)
	nx.set_node_attributes(G, mips, 'labels')
	A = nx.adjacency_matrix(G).toarray()
	dsd_A = calculator(A, 3)
	start = time.time()
	np.save("dsd.npy", dsd_A)
	end = time.time()
	print("Numpy save of DSD took " + str (end-start) + " seconds.")
	start = time.time()
	sc = SpectralClustering(1000, affinity='rbf', assign_labels='kmeans', n_jobs=-1, n_init=14)
	sc.fit(dsd_A)
	end = time.time()
	print ("Spectral clustering took " + str(end-start) + " seconds.")
	start = time.time()
	np.save("sc_labels.npy", sc.labels_)
	end = time.time()
	print ("Numpy save of clustering labels took " + str(end-start) + " seconds.")
	
	# train, test = split_data(mips, .7)

	# evalMetrics = getEvalMetrics()
	# print(evalMetrics['jaccard_score'](mips,train))
	
	# show_basic_attributes(G)
	# show_eda(mips)
