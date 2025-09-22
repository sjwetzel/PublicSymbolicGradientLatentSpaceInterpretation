# PublicSymbolicGradientLatentSpaceInterpretation

## Overview
This repository implements a framework for interpreting neural network latent spaces using symbolic search.

# Modify the following files to define system and model

## Functions that specify the system 
data_utils.py

## Specify the Neural Network
models.py

# Execute the following files in sequence to train and interpret the neural network.

## Create data
create_triplets.py

## Train the neural network
train_nn.py

## Extract latent space and gradient information from the network
latent_space_preparation

## Perform symbolic search
symbolic_nn_interpretation.jl
