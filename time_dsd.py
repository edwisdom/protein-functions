from ppi import calculator, read_network
import numpy as np
import networkx as nx


G = read_network(NETWORK_DATA)
A = nx.adjacency_matrix(G).toarray()

def time_dsd(step_counts):
	start = time.time()
	dsd_A = calculator(A, step_counts[0])
	end = time.time()
	print("Calculating DSD took " + str(end-start) + " seconds for 5 steps.")
	mean_changes = []
	for steps in step_counts[1:]:
		start = time.time()
		dsd_A_next = calculator(A, steps)
		end = time.time()
		print("Calculating DSD took " + str(end-start) + " seconds for " + str(steps) + " steps.")
		mean_changes.append(np.mean(np.abs(dsd_A_next - dsd_A)))
		dsd_A = dsd_A_next
	print(mean_changes)

time_dsd([5, 10, 20, 40, 80, -1])
