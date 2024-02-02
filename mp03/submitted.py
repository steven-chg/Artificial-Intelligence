'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!")

    # Idea: use a dictionary of counters (each word is a key, and the counter will keep track of occurrences of tags for each word)
    
    # declare an empty dictionary
    wordDictionary = {}

    # loop through the entire training data, sentence by sentence 
    for sentence in train:
        # loop through (word, tag) pair by pair
        for wordPair in sentence:
            # if the word has not been set as a key to a counter yet, initialize it
            if wordPair[0] not in wordDictionary:
                wordDictionary[wordPair[0]] = Counter()
            # for each (word, tag) pair, update the dictionary with the tag and count of 1
            wordDictionary[wordPair[0]].update({wordPair[1]: 1})

    ## at this point, wordDictionary is a dictionary of all words that appear in train as keys, and its value is a counter of occurrences of tags for that word
    
    # declare a 2D list of sentences with 'row' dimension as number of sentences in test data
    sentenceList = [[] for sentence in range (len(test))]

    ## find the tag used most often in the training set
    # declare a counter to combine all counters
    combinedCounter = Counter()
    # combine all counters by first retrieving all counters in the dictionary
    for counter in wordDictionary.values():
        combinedCounter.update(counter)
    # find the most frequent tag 
    mostCommonTag = max(combinedCounter, key = combinedCounter.get)

    # loop through the entire test data, sentence by sentence (by index)
    for sentenceIndex in range (len(test)):
        # loop through word by word
        for word in test[sentenceIndex]:
            # find what tag to place on the word 
            if word not in wordDictionary:
                # if the word is not in the wordDictionary, then set it with the most common tag
                sentenceList[sentenceIndex].append((word, mostCommonTag))
            else:
                # if the word is in the word is in the wordDictionary, then get the word's most common tag
                wordCommonTag = max(wordDictionary[word], key = wordDictionary[word].get)
                sentenceList[sentenceIndex].append((word, wordCommonTag))

    return sentenceList


def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # raise NotImplementedError("You need to write this part!")

    # declare an empty 2D list that will store the sentences with tags on the words ('row' dimension as number of sentences in test data)
    sentenceList = [[] for sentenceNum in range (len(test))]

    # Step 1: Count Occurrences of Tags, Tag Pairs, and Word/Tag Pairs
    # declare an empty dictionary that will track all tags and their number of occurrences
    tagOccurrenceDict = {}
    # declare an empty dictionary of counters that will track tag pairs and their number of occurrences (key is previous tag, value is a counter that tracks the tags that come after previous tag and their # of occurrences)
    tagPairOccurrenceDict = {}
    # declare an empty dictionary of counters that will track all word/tag pairs and their number of occurrences (key is the tag, value is a counter that tracks the words associated with the tag and their # of occurrences)
    wordTagPairOccurrenceDict = {}

    # loop through every sentence in training data set
    for sentence in train:
        # declare a variable to track the previous tag (initialize to empty string for every sentence)
        previousTag = ' '

        # loop through (word, tag) pair by pair
        for wordTagPair in sentence:
            # Step 1.1: Count Occurrencs of Tags 
            # if the tag has not been seen yet, initialize it
            if wordTagPair[1] not in tagOccurrenceDict:
                tagOccurrenceDict[wordTagPair[1]] = 1
            else:
                tagOccurrenceDict[wordTagPair[1]] += 1

            # Step 1.2: Count Occurrences of Tag Pairs
            # if the previous tag is not in the dictionary, then initialize it with a counter
            if previousTag not in tagPairOccurrenceDict:
                tagPairOccurrenceDict[previousTag] = Counter()
            # for each tag pair, update the dictionary with the tag that appears after the previous
            tagPairOccurrenceDict[previousTag].update({wordTagPair[1]: 1})
            # update the previous tag
            previousTag = wordTagPair[1]

            # LEADS TO TIMING OUT AS WE NEED TO LOOP WHEN FINDING PROBABILITY
            # # if the tag pair has not been seen yet, initialize it
            # if (wordTagPair[1], previousTag) not in tagPairOccurrenceDict:
            #     tagPairOccurrenceDict[(wordTagPair[1], previousTag)] = 1
            # else:
            #     tagPairOccurrenceDict[(wordTagPair[1], previousTag)] += 1
            

            # Step 1.3: Count Occurrences of Word/Tag Pairs
            # if the tag is not in the dictionary, then initialize it with a counter
            if wordTagPair[1] not in wordTagPairOccurrenceDict:
                wordTagPairOccurrenceDict[wordTagPair[1]] = Counter()
            # for each word tag pair, update the dictionary with the word that appears for the tag
            wordTagPairOccurrenceDict[wordTagPair[1]].update({wordTagPair[0]: 1})

            # LEADS TO TIMING OUT AS WE NEED TO LOOP WHEN FINDING PROBABILITY
            # # if the {word, tag} pair has not been seen yet, initialize it
            # if (wordTagPair[0], wordTagPair[1]) not in wordTagPairOccurrenceDict:
            #     wordTagPairOccurrenceDict[(wordTagPair[0], wordTagPair[1])] = 1
            # else:
            #     wordTagPairOccurrenceDict[(wordTagPair[0], wordTagPair[1])] += 1

    # sanity check
    # print(tagOccurrenceDict) 
    # print(tagPairOccurrenceDict)
    # print(wordTagPairOccurrenceDict)

    # Step 2 & 3: Compute Smooth Probabilities & Take Log of Each Probability
    # define a smoothing constant (testing 0.5)
    smoothingConstant = 0.5

    # Step 2.1 & 3.1 Compute Initial Probabilities P(Y1 = tag)
    # declare a dictionary to store initial probabilities of each tag; tag 'START' will have probability 1, all other tags have probability 0 (since sentences begin with 'START')
    initialProbabilities = {}
    # loop through all tags
    for tag in tagOccurrenceDict.keys():
        if tag == 'START':
            initialProbabilities[tag] = math.log((tagOccurrenceDict[tag] + smoothingConstant) / (len(train) + smoothingConstant*(len(tagOccurrenceDict) + 1)))
        else:
            initialProbabilities[tag] = math.log((smoothingConstant) / (len(train) + smoothingConstant*(len(tagOccurrenceDict) + 1)))

    # Step 2.2 & 3.2 Compute Transition Probabilities P(tagB|tagA)
    # declare a dictionary to store transition probabilities of each tag pair; (tagB, tagA) where tagB follows tagA
    transitionProbabilities = {}
    # loop through all previous tags
    for previousTag in tagPairOccurrenceDict.keys():
        # for each previous tag, loop through all tags that could follow
        for nextTag in tagPairOccurrenceDict[previousTag]:
            # create the tag pair (nextTag, previousTag)
            tagPair = (nextTag, previousTag)
            transitionProbabilities[tagPair] = math.log((tagPairOccurrenceDict[previousTag].get(nextTag, 0) + smoothingConstant) / (sum(tagPairOccurrenceDict[previousTag].values()) + smoothingConstant*(len(tagPairOccurrenceDict[previousTag]) + 1)))
            # ORIGINAL INEFFICIENT METHOD
            # transitionProbabilities[tagPair] = math.log((tagPairOccurrenceDict[tagPair] + smoothingConstant) / (sum(value for key, value in tagPairOccurrenceDict.items() if key[1] == tagPair[1]) + smoothingConstant*(sum(1 for key in tagPairOccurrenceDict if key[1] == tagPair[1]) + 1)))

    # Step 2.3 & 3.3 Compute Emission Probabilities P(word|tag)
    # declare a dictionary to store emission probabilities 
    emissionProbabilities = {}
    # loop through all tags
    for tag in wordTagPairOccurrenceDict.keys():
        # loop through all words associated with the tag
        for word in wordTagPairOccurrenceDict[tag]:
            # create the word tag pair (word, tag)
            wordTagPair = (word, tag)
            emissionProbabilities[wordTagPair] = math.log((wordTagPairOccurrenceDict[tag].get(word, 0) + smoothingConstant) / (sum(wordTagPairOccurrenceDict[tag].values()) + smoothingConstant*(len(wordTagPairOccurrenceDict[tag]) + 1)))
            # ORIGINAL INEFFICIENET METHOD
            # emissionProbabilities[wordTagPair] = math.log((wordTagPairOccurrenceDict[wordTagPair] + smoothingConstant) / (sum(value for key, value in wordTagPairOccurrenceDict.items() if key[1] == wordTagPair[1]) + smoothingConstant*(sum(1 for key in wordTagPairOccurrenceDict if key[1] == wordTagPair[1]) + 1)))

    # sanity check
    # print(initialProbabilities)
    # print(transitionProbabilities)
    # print(emissionProbabilities)

    # Step 4: Construct the Trellis (row dimension = total # of all tags, column dimension = total # of words in a sentence)


    # Step 5: Return Best Path Through Trellis


def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



