# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4

    # useful metrics
    numSamples = np.shape(train_set)[0]
    numFeatures = np.shape(train_set)[1]

    # initialize the learning rate as well as weight matrix and bias (to 0)
    learningRate = 0.01
    W = np.zeros(numFeatures) 
    b = 0

    # loop through entire train_set for max_iter times to train W and b based on the learning rate
    for iter in range(max_iter):
        # loop through each sample in the training set
        for sampleNum in range(numSamples):
            # for each sample, 1) find the classifier output
            result = np.sum(W*train_set[sampleNum]) + b
            if result > 0: classifierOutput = 1
            else: classifierOutput = 0
            # 2) update the weight vectors based on classifier output and actual output/labels 
            if (classifierOutput != train_labels[sampleNum]) and train_labels[sampleNum] == 1: 
                W += learningRate*train_set[sampleNum]
                b += learningRate
            elif (classifierOutput != train_labels[sampleNum]) and train_labels[sampleNum] == 0: 
                W -= learningRate*train_set[sampleNum]
                b -= learningRate

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4

    # Note:
    # np.shape(train_set) = (# of samples, # of features)
    # np.shape(train_labels) = (# of features), where each value can be 0 to (# of features - 1)

    # Step 1: Train the Perceptron
    WTrained, bTrained = trainPerceptron(train_set, train_labels, max_iter)

    # Step 2: Find Label for Dev_Set
    # declare an empty list for storing dev_labels
    dev_labels = []
    # loop through each sample in dev_set
    for sampleNum in range(np.shape(dev_set)[0]):
        result = np.sum(WTrained*dev_set[sampleNum]) + bTrained
        # if result is > 0, classify 1; else, classify 0
        if result > 0: dev_labels.append(1)
        if result <= 0: dev_labels.append(0)

    return dev_labels



