README for dataset FIRSTMM_DB


=== Introduction ===

This folder contains 1 dataset of undirected labeled and attributed graphs for 
graph classification: FIRSTMM_DB.


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DD_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description of the dataset === 

Node labels:

 1 'bottom'
 2 'middle'
 3 'top'
 4 'handle'
 5 'usable_are'

Node attribute:
  [first dimension]:
  [change in curvature]
	

Edge attributes:
  [first dimension, second dimension]:	
  [distance between points, change in curvature]


Additional information: 

FIRSTMM_DB_graph_list.txt: list of object names 
(each line corresponds to one object in the same order as in graph_labels)

FIRSTMM_DB_coordinates.txt: 3D coordinates

FIRSTMM_DB_normals.txt: 3D normals (unit length); contains NaNs


=== Previous Use of the Dataset ===

Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph 
Kernels from Propagated Information. Under review at MLJ.

M. Neumann, P. Moreno, L. Antanas, R. Garnett, K. Kersting. Graph Kernels for 
Object Category Prediction in Task-Dependent Robot Grasping. Eleventh Workshop 
on Mining and Learning with Graphs (MLG-13), Chicago, Illinois, USA, 2013.

The original object database can be found here:
http://www.first-mm.eu/data.html

It is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License 
(http://creativecommons.org/licenses/by-sa/3.0/). 


