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

Here, I will outline some of the major model parameters that I iteratively tweaked, along with the effect on the model's accuracy rate.

### Baseline Model

I began with a simple model:
- Embedding layer with an input of pre-trained 50-dimensional vectors (GloVe 6B.50D)
- Bidirectional LSTM of size 50, with dropout 0.1
- Pooling layer (average + max concatenated)
- FC layer of size 25, with dropout 0.1
- Output FC layer of size 6 (one per class)

I used a batch size of 32 and the Adam optimizer, which is an alternative to stochastic gradient descent. Each parameter of the network has a separate learning rate, which are continually adapted as the network learns. For more on the Adam optimizer, read [here](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). 

**Keras Output:**

```
loss: 0.0447 - acc: 0.9832 - val_loss: 0.0472 - val_acc: 0.9824
```

The model had trained for 2 epochs. This output from Keras shows loss and accuracy on the training data used, and the 10% held out for cross-validation. Since the out-of-sample predictions best indicate how well the model generalizes, the val_loss and val_acc will be the measures that I report for future model iterations.

### Batch Size / Epochs

First, I recognized that I could train for multiple epochs. The network eventually overfits if we add too many epochs, so first, we can add a callback to stop early. In this code, if val_loss doesn't improve after 3 epochs, the model stops training. 

```python
es = EarlyStopping(monitor='val_loss',
                   min_delta=0,
                   patience=3,
                   verbose=0, mode='auto')
```

Moreover, we can save the best model with the following callback:

```python
best_model = 'models/model_filename.h5'
checkpoint = ModelCheckpoint(best_model, 
                             monitor='val_loss', 
                             verbose=0, 
                             save_best_only=True, mode='auto')
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, callbacks=[es, checkpoint], validation_split=0.1)
```

Second, I migrated the network to my GPU by downloading CUDA and Tensorflow-GPU. This allowed me to change the batch size to 1024 and train my network much faster.

**Loss: 0.0454, Accuracy: 0.9832**

### Dropout 

I wanted to check if I could tune the model's performance by increasing dropout before experimenting with the network's architecture. Almost all of these efforts, done alone, actually lowered the model accuracy.

- Embedding Layer Dropout to 0.2 -- Loss: 0.0469, Accuracy: 0.9827
- Final Layer Dropout to 0.3 -- Loss: 0.0482, Accuracy: 0.9831
- LSTM Dropout to 0.3 -- Loss: 0.0473, Accuracy: 0.9827
- Recurrent Dropout to 0.3 -- Loss: 0.0465, Accuracy: 0.9831

These results make some sense in hindsight, since the network size is relatively small. As [this paper found](https://pdfs.semanticscholar.org/3061/db5aab0b3f6070ea0f19f8e76470e44aefa5.pdf), applying dropout in the middle and after LSTM layers tends to worsen performance. This, of course, didn't explain why increasing dropout in the embedding layer (which comes before the LSTM) worsened performance.

As I found in this [paper on CNNs](https://arxiv.org/pdf/1411.4280.pdf), dropping random weights doesn't actually help when there is spatial correlation in the feature maps. Since natural language also exhibits spatial/sequential correlation, spatial dropout would be a much better choice, since it drops out entire feature maps. After adding a spatial dropout of 0.2 before the LSTM layer, the network finally improved.

**Loss: 0.0452, Accuracy: 0.9834**

### Architecture

First, I experimented with a different RNN cell. I simply reconstructed the previous network's architecture, but replaced LSTM cells with GRU cells. GRU layers only have two gates, a reset and update gate -- whereas the update gate encodes the previous inputs ("memory"), the reset gate combines the input with this memory. 

Whereas the LSTM can capture long-distance connections due to its hidden state, this may not be necessary for identifying toxicity, since a comment is likely to be toxic throughout. For more on the difference between GRUs and LSTMs, read [here](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/). For an evaluation of the two on NLP tasks, see [this paper](https://arxiv.org/pdf/1412.3555v1.pdf). 

Surprisingly, the GRU performed comparably to the LSTM without any further tuning.

**Loss: 0.0450, Accuracy: 0.9832**

Second, I used larger pre-trained embedding vectors (from 50 dimensions to 300). Furthermore, I increased the number of words that the model was using for each comment in increments of 50, going up from 100 originally to 300 once model performance stopped improving. This simple change improved performance significantly for the LSTM.

**Loss: 0.0432, Accuracy: 0.09838**

Third, and perhaps most importantly, I added a convolutional layer of size 64, with a window size of 3, in between the recurrent and FC layers for both the LSTM and GRU network. Although I found RCNNs rather late in my model iteration process, I've explained them above in the [Background Research section](https://github.com/edwisdom/toxic-comments#recurrent-convolutional-neural-networks).

**LSTM - Loss: 0.0412, Accuracy: 0.9842**

**GRU  - Loss: 0.0414, Accuracy: 0.9842**

Finally, I decided to stack another convolutional layer of size 64, with window size 6, before the FC layer for both networks. I also tried to add a FC layer of size 64 before the output layer. Both of these slightly improved the model, although the GRU benefitted more from the additional convolution, whereas the LSTM benefitted more from the added FC layer.


| Loss By Model | CNN Layer | FC Layer |
|:-------------:|:---------:|:--------:|
| GRU           | 0.0406    | 0.0408   |
| LSTM          | 0.0411    | 0.0402   |


Ensembled together, the two best-performing networks here reach **98.48% accuracy**.

### Other Things I Learned That Don't Deserve a Whole Section 

- Learning Rate Optimizers: For my data and model, Adam vastly outperformed both Adadelta, Adagrad, and RMSProp. I include a more thorough comparison of the optimizers below, from Suki Lau on Towards Data Science.

![alt text](https://cdn-images-1.medium.com/max/800/1*OjcTfMw6dmOmP4lRE7Ud-A.jpeg)

- Alternative Embeddings: All the figures I present here use the GloVe vectors, but I also tried to use pre-trained FastText vectors of the same size (300D), and the network performed comparably. 

## Future Work

Here are some things that I did not get to tune that would make for interesting results:

1. Using only max-pooling layers vs. using only average-pooling layers vs. using both
2. Initializing different learning rates and setting a decay rate
3. Different activation functions -- Tanh vs. PreLu vs. ReLu
4. More convolutional layers with larger window sizes to capture long-distance connections
5. Preprocessing comments using NLP techniques such as lemmatization, removing stop words, etc.

## Credits

I would like to thank Prof. Liping Liu, Daniel Dinjian, and Nathan Watts for thinking through problems with me and helping me learn the relevant technologies faster. 

I got the data for this model from a [Kaggle competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), and I was helped greatly by [this exploratory data analysis by Jagan Gupta](https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda).

