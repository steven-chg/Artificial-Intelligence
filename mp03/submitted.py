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
    raise NotImplementedError("You need to write this part!")


def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



