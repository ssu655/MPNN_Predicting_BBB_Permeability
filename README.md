# MPNN_Predicting_BBB_Permeability
This project is a pipeline of training a Graph Neural Network to predict the physiochemical property of small drug molecules, based on their molecular graphs. 

A quick demo of the project is in the Demo.ipynb file. The computation is all very quick to run, so feel free to download and play with it.

Message Passing Neural Network is a special type of Graph Neural Networks that utilizes both node features AND edge features, which makes it suitable for differentiating and learning the various chemical bonds in molecular graphs. Message passing describes the activity of passing the imformation of all neighbors of a node to this node, which is an act of aggregation, and similar to convolution. 

For the dataset, we use PAMPA assay, which is a reliable predictor of blood brain barrier (BBB) permeability for small molecules, (article https://doi.org/10.3389/fphar.2023.1291246). From PubChem database, we obtain a dataset of the PAMPA measurement of 438 small molecules and their SMILES descriptors (PubChem AID: 1845228). 

