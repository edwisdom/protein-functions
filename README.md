# Protein Function Prediction with Graph Clustering

This project uses a novel graph distance measure and spectral clustering in order to predict protein functions just from protein-interaction networks and some known labels. 

By improving automated annotation of protein functions, we can close the gap between the rapidly accumulating biological sequence data and experimentally annotated proteins. Moreover, since diseases are often caused by changes in protein function, this information could help guide treatment research. 

## Getting Started

These instructions will allow you to run this project on your local machine. 

### Install Requirements

Once you have a virtual environment in Python, you can simply install necessary packages with: `pip install requirements.txt`

Some of these requirements may not be completely necessary, as they may have been removed or replaced by the time the project was finished. We apologize for the extra space needed.

### Clone This Repository

```
git clone https://github.com/edwisdom/protein-functions
```

### Run Script

Run the script with Python to see the results printed out to your terminal or IDE:

```
python ppi.py
```

## Background Research

This section covers some of our basic research in protein interaction networks and graph-based clustering methods.

### Protein-Protein Interaction Networks

In the past few years, sequencing technology has given us data on a number of organisms' genome. However, interpreting this data requires understanding protein function, and experimental annotation simply cannot keep up (see Figure 1). Therefore, protein-protein interaction networks have become increasingly important in predicting function, since network distance highly correlates with functional similarity (see Figure 2).

<figure>
    <img src='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1847944/bin/msb4100129-f1.jpg' width="600"/>
    <font size="2">
    <figcaption> Figure 1: Percentage of annotated vs. unannotated proteins by species, from Sharan, Ulitsky, and Shamir 
    </figcaption>
    </font>
</figure>

<figure>
    <img src='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1847944/bin/msb4100129-f3.jpg' />
    <font size="2">
    <figcaption> Figure 2: Correlation between protein functional similarity and network distance, also from Sharan, Ulitsky, and Shamir 
    </figcaption>
    </font>
</figure>


\
For more on PPI networks, read:
- [Introduction to Protein Function Prediction for Computer Scientists](http://biofunctionprediction.org/cafa-targets/Introduction_to_protein_prediction.pdf)
- [Network-Based Prediction of Protein Function ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1847944/)
- [Review of ML Methods for Protein Function Prediction](https://www.frontiersin.org/articles/10.3389/fphys.2016.00075/full)

### Graph-Based Clustering

In machine learning terms, this problem boils down to a graph-based semi-supervised multi-class classification problem. This has often been approached with clustering, and spectral clustering in particular. Spectral graph theory, an [active field of interest](https://arxiv.org/pdf/1609.08072.pdf), fundamentally involves constructing a Laplacian matrix, finding its eigenvalues, and relating those to some properties of the graph. 


- [A Tutorial on Spectral Clustering](https://arxiv.org/pdf/0711.0189.pdf)
- [Graph-Based Semi-Supervised Learning Methods](http://www.cs.cmu.edu/afs/cs/Web/People/frank/papers/thesis.pdf)
- [Survey of Graph Clustering Algorithms](http://snap.stanford.edu/class/cs224w-2014/projects2014/cs224w-21-final.pdf)

## Data

The data here consists of over 5,000 proteins, over 60,000 edges (thereby making a sparse graph), and over 4,000 label instances. The edges between the proteins represent physical contact, and the labels represent known experimental functional annotations. For more information about the dataset, especially the biological meaning of the functional annotations, see [Tufts University Professor Lenore Cowen's website](http://dsd.cs.tufts.edu/).

### Label Imbalances

Note that out of the 18 labels, some functional annotations (01, 42) are much more common than others (38, 41). This class imbalance can often bias clustering or classification algorithms towards the majority or plurality class. Unlike cases where we want to detect anomalies like negative sentiment in NLP or cancerous cells in image classification, in this application, the problem is less acute because the detection of some labels isn't inherently more valuable. For more on the class-imbalance problem, see [this paper](https://link.springer.com/article/10.1007/s13748-016-0094-0).

![alt text](https://github.com/edwisdom/protein-functions/blob/master/freq_labels.png "Label Imbalance in Protein-Protein Interaction Data")

### Multi-Label Frequencies

Some proteins have multiple labels, thus complicating our task, since most clustering algorithms perform best when they have to [partition the data](https://link.springer.com/article/10.1007/s40745-015-0040-1) into separate clusters. Although the number of proteins goes down as we increase the number of labels, the majority of our data does have multiple labels. 

![alt text](https://github.com/edwisdom/protein-functions/blob/master/freq_num_labels.png "Multi-Label Frequency in PPI Data")


### Multiple Correlated Labels

As the following heatmap shows, there is significant overlap between some functional labels and others. A curious finding was that "#" does not correlate with any other labels -- this is because its biological annotation in the BioGRID database is "unclear classification." 

![alt text](https://github.com/edwisdom/protein-functions/blob/master/corr_heatmap.png "Protein Function Correlations Heatmap")

### A Scale-Free, Non-Random Network

The PPI network is [scale-free](http://rakaposhi.eas.asu.edu/cse494/scalefree.pdf), exhibiting a pattern where a degree and its frequency in the network is inversely proportional. In other words, a few nodes have very high degree, whereas most do not. This also means that the average shortest path in the network will be relatively small, making it difficult to use conventional distance measures as a notion of similarity between nodes.

![alt text](https://github.com/edwisdom/protein-functions/blob/master/degree_distribution.png "Degree Distribution of PPI Network")

## Model

Here, we outline the major components of the model. 

### Diffusion-State Distance (DSD)

The work on a diffusion-based distance metric comes from Tufts' Bioinformatics and Computational Biology Research Group. Because the average shortest path is low across the network due to its degree distribution, the DSD uses random walks in order to normalize the pairwise distances between vertices. The key insight is that paths through high-degree nodes should be valued less -- for example, two people liking Beyonce does not mean they are in close social circles. For a more in-depth discussion of the metric, see [the original paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0076339).  

<figure>
    <img src='https://github.com/edwisdom/protein-functions/blob/master/shortest_paths.png'/>
    <font size="2">
    <figcaption> Figure 3: Distribution of shortest path distances in the yeast PPI network 
    </figcaption>
    </font>
</figure>

<figure>
    <img src='https://github.com/edwisdom/protein-functions/blob/master/dsd_paths.png' />
    <font size="2">
    <figcaption> Figure 4: Distribution of DSD distances in the yeast PPI network
    </figcaption>
    </font>
</figure>


### Gaussian Kernel

In order to convert the resulting distance matrix from the DSD into an affinity matrix, we use the standard Gaussian RBF kernel. This gives us a matrix of size |V²|, with each entry aᵢⱼ corresponding to the similarity between vertices vᵢ and vⱼ. 


```python
    delta = 1
    rbf_matrix = np.exp(- dsd_A ** 2 / (2. * delta ** 2))
```


### Spectral Clustering

Once we get an affinity matrix, we use spectral clustering from scikit-learn to create a low-dimensional embedding, on which we apply the k-means algorithm to produce clusters. More specifically, we recursively use spectral clustering on groups of size > 100, and throw out any clusters with less than 3 nodes. 

Then, in each cluster, we simply look at the frequencies of labels that we know from our training data. If those frequencies are above a certain threshold, then we predict that label for all of the nodes in that cluster. 


## Evaluation

We report an average of metrics run on a training/test split of 0.7/0.3. 

|     Metric     |     Value     |
|:--------------:|:-------------:|
|   Rand Index   |   0.854       | 
| Precision      |   0.207       |
| Recall         |   0.053       |
| F1 Measure     |   0.158       |
| Jaccard Index  |   0.044       |
| Hamming Loss   |   0.146       |

The Hamming loss metric, commonly used for multi-label classification problems, shows us that we get labels wrong approximately *15% of the time*. This is reasonable, although our F1 measure of 0.158 pales in comparison to the top submissions for [previous protein function prediction competitions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5015320/), which score over 0.3.

We believe this is likely because we have yet to tune important parameters of our model, which we will discuss in the next section. 


## Future Work

Here are some parameters that we did not get to tune that would make for interesting results:

1. Changing the number of clusters (k): Currently, we divide the number of nodes by 5 to determine the number of clusters, though this may be too many clusters for scikit-learn, which "works well for a small number of clusters" according to [the documentation](http://scikit-learn.org/stable/modules/clustering.html).
2. Using different values of delta for the RBF kernel: We default to using delta=1, however, modifying this kernel parameter could significantly improve results. [Prior work with SVMs](https://arxiv.org/pdf/1507.06020.pdf) has shown that performance can be very sensitive to choice of this parameter. 
3. Number of runs through K-means algorithm: According to scikit-learn, the k-means algorithm can produce noisy clusters, so we run it with 10 different centroid seeds and output the best clustering in terms of inertia. However, it may be necessary to increase the number of runs through k-means to produce a better clustering.
4. Using second- and third-class MIPS labels: Once we can tune these model parameters, it may be informative to try to classify second and third-level protein labels to see how increasing the number of possible clusters impacts performance. 

## Credits

We would like to thank Prof. Liping Liu, Prof. Lenore Cowen, and Nathan Watts for helping us think through our architecture and working through knotty technical problems with us.

### Section Breakdown

**Edwin**: Organized the README, wrote functions to read in and organize the data, made plots for exploratory data analysis, created evaluation metrics, and figured out how to deal with scikit-learn's spectral clustering. 

**Daniel**: Wrote the function that recursively performs spectral clustering, implemented the label prediction from these clusters, computed the necessary values to perform evaluation, helped figure out spectral clustering, and cleaned up EDA and un-modular code. 
