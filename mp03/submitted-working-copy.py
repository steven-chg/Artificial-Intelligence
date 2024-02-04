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

    # declare empty sets to store all words and all tags and all word tag pairs that occur in the test data set (for ease of constructing trellis)
    wordSet = set()
    tagSet = set()
    wordTagPairSet = set()

    # Step 1: Count Occurrences of Tags, Tag Pairs, and Word/Tag Pairs
    # declare an empty default dictionary that will track all tags and their number of occurrences
    tagOccurrenceDict = defaultdict(int)
    # declare an empty default dictionary of counters that will track tag pairs and their number of occurrences (key is previous tag, value is a counter that tracks the tags that come after previous tag and their # of occurrences)
    tagPairOccurrenceDict = defaultdict(Counter)
    # declare an empty default dictionary of counters that will track all word/tag pairs and their number of occurrences (key is the tag, value is a counter that tracks the words associated with the tag and their # of occurrences)
    wordTagPairOccurrenceDict = defaultdict(Counter)

    # loop through every sentence in training data set
    for sentence in train:
        # declare a variable to track the previous tag (initialize to empty string for every sentence)
        previousTag = ' '

        # loop through (word, tag) pair by pair
        for wordTagPair in sentence:
            # add the word, tag, and (word, tag) pair into the set
            wordSet.add(wordTagPair[0])
            tagSet.add(wordTagPair[1])
            wordTagPairSet.add((wordTagPair[0], wordTagPair[1]))

            # Step 1.1: Count Occurrencs of Tags 
            tagOccurrenceDict[wordTagPair[1]] += 1

            # Step 1.2: Count Occurrences of Tag Pairs
            # if the previous tag is not in the dictionary (excluding ' ' & also does not include 'END' since it is the last tag), then initialize it with a counter 
            # for each tag pair, update the dictionary with the tag that appears after the previous
            if previousTag != ' ':
                tagPairOccurrenceDict[previousTag][wordTagPair[1]] += 1
            # update the previous tag
            previousTag = wordTagPair[1]
            
            # Step 1.3: Count Occurrences of Word/Tag Pairs
            # for each word tag pair, update the dictionary with the word that appears for the tag
            wordTagPairOccurrenceDict[wordTagPair[1]][wordTagPair[0]] += 1

    # Step 2 & 3: Compute Smooth Probabilities & Take Log of Each Probability
    # define a smoothing constant (testing 10^-5)
    smoothingConstant = 1e-5

    # Step 2.1 & 3.1 Compute Initial Probabilities P(Y1 = tag)
    # declare a dictionary to store initial probabilities of each tag; tag 'START' will have probability 1, all other tags have probability 0 (since sentences begin with 'START')
    initialProbabilities = {}
    # loop through all tags
    for tag in tagOccurrenceDict.keys():
        tagCount = tagOccurrenceDict[tag]
        numberOfSentences = len(train)
        numberOfDiffTags = len(tagOccurrenceDict)
        if tag == 'START':
            initialProbabilities[tag] = math.log((tagCount + smoothingConstant) / (numberOfSentences + smoothingConstant*(numberOfDiffTags + 1)))
        else:
            initialProbabilities[tag] = math.log((smoothingConstant) / (numberOfSentences + smoothingConstant*(numberOfDiffTags + 1)))

    # Step 2.2 & 3.2 Compute Transition Probabilities P(tagB|tagA)
    # declare a dictionary to store transition probabilities of each tag pair; (tagB, tagA) where tagB follows tagA
    transitionProbabilities = {}
    # loop through all tags
    for previousTag in tagSet:
        # find total number of occurrences of tag pairs with previousTag as the previous tag
        if previousTag != 'END':
            totalPreviousTag = sum(tagPairOccurrenceDict[previousTag].values())
        # find count of tags that follow previous tag (number of nextTags with previousTag as the previous tag)
        # tagsFollowing = len(tagPairOccurrenceDict[previousTag])
        # loop through all tags again, to find all possible combination of tags
        for nextTag in tagSet:
            # find probabilities of actual tag pairs that have appeared in the training data set (first condition will allow all tags excluding 'END' tag)
            if previousTag in tagPairOccurrenceDict.keys() and nextTag in tagPairOccurrenceDict[previousTag]:
                # create the tag pair (nextTag, previousTag)
                tagPair = (nextTag, previousTag)
                transitionProbabilities[tagPair] = math.log((tagPairOccurrenceDict[previousTag].get(nextTag, 0) + smoothingConstant) / (totalPreviousTag + smoothingConstant*(len(tagSet) + 1)))
            # find probabilities of tag pairs that have NOT appeared in the training data set (previousTag could be in tagPairOccurrence as a key, but the next tag might not be paired with it)
            elif (previousTag not in tagPairOccurrenceDict.keys() or nextTag not in tagPairOccurrenceDict[previousTag]) and previousTag != 'END':  
                transitionProbabilities[(nextTag, previousTag)] = math.log(smoothingConstant / (totalPreviousTag + smoothingConstant*(len(tagSet) + 1)))
            elif (previousTag not in tagPairOccurrenceDict.keys() or nextTag not in tagPairOccurrenceDict[previousTag]) and previousTag == 'END':
                transitionProbabilities[(nextTag, previousTag)] = math.log(smoothingConstant / (smoothingConstant*len(tagSet)))
                
    # Step 2.3 & 3.3 Compute Emission Probabilities P(word|tag)
    # declare a dictionary to store emission probabilities 
    emissionProbabilities = {}
    # loop through all tags
    for tag in tagSet:
        totalOccurrenceWithTag = sum(wordTagPairOccurrenceDict[tag].values())
        # loop through all words
        for word in wordSet:
            # find probabilities of actual word, tag pairs that have appeared in the training data set
            if tag in wordTagPairOccurrenceDict.keys() and word in wordTagPairOccurrenceDict[tag]:
                # create the word tag pair (word, tag)
                wordTagPair = (word, tag)
                wordTagCount = wordTagPairOccurrenceDict[tag].get(word, 0)
                emissionProbabilities[wordTagPair] = math.log((wordTagCount + smoothingConstant) / (totalOccurrenceWithTag + smoothingConstant*(len(wordSet) + 1)))
            # find probabilities of word tag pairs that haven't appeared, but the words and tags we have seen in the dat set
            elif tag not in wordTagPairOccurrenceDict.keys() or word not in wordTagPairOccurrenceDict[tag]:
                emissionProbabilities[(word, tag)] = math.log(smoothingConstant / (totalOccurrenceWithTag + smoothingConstant*(len(wordSet)) + 1))
        # add an ('UNKNOWN', tag) for words that were not seen in the training data for each tag 
        unknownWordTagPair = ('UNKNOWN', tag)
        emissionProbabilities[unknownWordTagPair] = math.log(smoothingConstant / (totalOccurrenceWithTag + smoothingConstant*(len(wordSet) + 1)))

    # Step 4: Construct the Trellis (row dimension = total # of all tags, column dimension = total # of words in a sentence)    AND     Step 5: Return Best Path Through Trellis
    # declare a 2D list for storing the sentences in the test data with tags
    sentenceList = [[] for sentenceNum in range (len(test))]
    # loop through each sentence index in the test data set and construct a 2D list for the trellis for each sentence and also a backpointer 2D list to store pointer to previous tag/time pair in the path
    for sentenceNum in range (len(test)):
        trellis = [[0]*len(test[sentenceNum]) for row in range(len(tagSet))]
        backPointers = {tag: [0] * len(test[sentenceNum]) for tag in tagSet}

        # initialization step: loop through the entire first column and calculate prior probability and emission probability
        for tagNum, tag in enumerate(tagSet):
            firstWord = test[sentenceNum][0]    
            trellis[tagNum][0] = initialProbabilities[tag]+emissionProbabilities[(firstWord, tag)]
            backPointers[tag][0] = 0
        
        # recursion step: loop through each time step (or word in the sentence; starting from 2nd word)
        for wordNum in range (1, len(test[sentenceNum])):
            word = test[sentenceNum][wordNum]
            # loop through each state/tag 
            for tagNum, tag in enumerate(tagSet):
                # check if the word is in the wordList, if so, proceed as normal
                if word in wordSet:
                    # find the maximum probability (from current word to next word) and the previous state 
                    maximumProbability, previousStateTag = max((trellis[previousTagNum][wordNum - 1]+transitionProbabilities[(tag, previousTag)]+emissionProbabilities[(word, tag)], previousTag) for previousTagNum, previousTag in enumerate(tagSet))
                else:
                    maximumProbability, previousStateTag = max((trellis[previousTagNum][wordNum - 1]+transitionProbabilities[(tag, previousTag)]+emissionProbabilities[('UNKNOWN', tag)], previousTag) for previousTagNum, previousTag in enumerate(tagSet))
                # store maximum probability into the trellis and previousState (tag number of previous state) into the backpointer list
                trellis[tagNum][wordNum] = maximumProbability
                backPointers[tag][wordNum] = previousStateTag

        # termination step: find the best path probability and the best path pointer (tagNum of the final tag)
        bestpathprob, bestpathtag = max((trellis[tagNum][len(test[sentenceNum]) - 1], tag) for tagNum, tag in enumerate (tagSet))

        # backtracking step: starting from the final word, go back to find the previous best tag, inserting it to the front
        for wordNum in reversed(range(len(test[sentenceNum]))):
            sentenceList[sentenceNum].insert(0, (test[sentenceNum][wordNum], bestpathtag))
            bestpathtag = backPointers[bestpathtag][wordNum]

    return sentenceList

    # # declare an empty list to store all words and all tags and all word tag pairs that occur in the test data set (for ease of constructing trellis)
    # wordList = []
    # tagList = []
    # wordTagPairList = []

    # # Step 1: Count Occurrences of Tags, Tag Pairs, and Word/Tag Pairs
    # # declare an empty dictionary that will track all tags and their number of occurrences
    # tagOccurrenceDict = {}
    # # declare an empty dictionary of counters that will track tag pairs and their number of occurrences (key is previous tag, value is a counter that tracks the tags that come after previous tag and their # of occurrences)
    # tagPairOccurrenceDict = {}
    # # declare an empty dictionary of counters that will track all word/tag pairs and their number of occurrences (key is the tag, value is a counter that tracks the words associated with the tag and their # of occurrences)
    # wordTagPairOccurrenceDict = {}

    # # loop through every sentence in training data set
    # for sentence in train:
    #     # declare a variable to track the previous tag (initialize to empty string for every sentence)
    #     previousTag = ' '

    #     # loop through (word, tag) pair by pair
    #     for wordTagPair in sentence:
    #         # if the word is new, add it to the list
    #         if wordTagPair[0] not in wordList:
    #             wordList.append(wordTagPair[0]) 
    #         # if the tag is new, add it to the list
    #         if wordTagPair[1] not in tagList:
    #             tagList.append(wordTagPair[1])
    #         # if the word tag pair is new, add it to the list
    #         if (wordTagPair[0], wordTagPair[1]) not in wordTagPairList:
    #             wordTagPairList.append((wordTagPair[0], wordTagPair[1]))

    #         # Step 1.1: Count Occurrencs of Tags 
    #         # if the tag has not been seen yet, initialize it
    #         if wordTagPair[1] not in tagOccurrenceDict:
    #             tagOccurrenceDict[wordTagPair[1]] = 1
    #         else:
    #             tagOccurrenceDict[wordTagPair[1]] += 1

    #         # Step 1.2: Count Occurrences of Tag Pairs
    #         # if the previous tag is not in the dictionary (excluding ' '), then initialize it with a counter 
    #         if previousTag not in tagPairOccurrenceDict and previousTag != ' ' :
    #             tagPairOccurrenceDict[previousTag] = Counter()
    #         # for each tag pair, update the dictionary with the tag that appears after the previous
    #         if previousTag != ' ':
    #             tagPairOccurrenceDict[previousTag].update({wordTagPair[1]: 1})
    #         # update the previous tag
    #         previousTag = wordTagPair[1]

    #         # LEADS TO TIMING OUT AS WE NEED TO LOOP WHEN FINDING PROBABILITY
    #         # # if the tag pair has not been seen yet, initialize it
    #         # if (wordTagPair[1], previousTag) not in tagPairOccurrenceDict:
    #         #     tagPairOccurrenceDict[(wordTagPair[1], previousTag)] = 1
    #         # else:
    #         #     tagPairOccurrenceDict[(wordTagPair[1], previousTag)] += 1
            
    #         # Step 1.3: Count Occurrences of Word/Tag Pairs
    #         # if the tag is not in the dictionary, then initialize it with a counter
    #         if wordTagPair[1] not in wordTagPairOccurrenceDict:
    #             wordTagPairOccurrenceDict[wordTagPair[1]] = Counter()
    #         # for each word tag pair, update the dictionary with the word that appears for the tag
    #         wordTagPairOccurrenceDict[wordTagPair[1]].update({wordTagPair[0]: 1})

    #         # LEADS TO TIMING OUT AS WE NEED TO LOOP WHEN FINDING PROBABILITY
    #         # # if the {word, tag} pair has not been seen yet, initialize it
    #         # if (wordTagPair[0], wordTagPair[1]) not in wordTagPairOccurrenceDict:
    #         #     wordTagPairOccurrenceDict[(wordTagPair[0], wordTagPair[1])] = 1
    #         # else:
    #         #     wordTagPairOccurrenceDict[(wordTagPair[0], wordTagPair[1])] += 1

    # # sanity check
    # # print(tagOccurrenceDict) 
    # # print(tagPairOccurrenceDict)
    # # print(wordTagPairOccurrenceDict)

    # # Step 2 & 3: Compute Smooth Probabilities & Take Log of Each Probability
    # # define a smoothing constant (testing 0.5)
    # smoothingConstant = 0.5

    # # Step 2.1 & 3.1 Compute Initial Probabilities P(Y1 = tag)
    # # declare a dictionary to store initial probabilities of each tag; tag 'START' will have probability 1, all other tags have probability 0 (since sentences begin with 'START')
    # initialProbabilities = {}
    # # loop through all tags
    # for tag in tagOccurrenceDict.keys():
    #     if tag == 'START':
    #         initialProbabilities[tag] = math.log((tagOccurrenceDict[tag] + smoothingConstant) / (len(train) + smoothingConstant*(len(tagOccurrenceDict) + 1)))
    #     else:
    #         initialProbabilities[tag] = math.log((smoothingConstant) / (len(train) + smoothingConstant*(len(tagOccurrenceDict) + 1)))

    # # COMBINE INTO A SINGLE DOUBLE FOR LOOP GOING THROUGH ALL THE TAGS; check if it is a tag pair
    # # Step 2.2 & 3.2 Compute Transition Probabilities P(tagB|tagA)
    # # declare a dictionary to store transition probabilities of each tag pair; (tagB, tagA) where tagB follows tagA
    # transitionProbabilities = {}
    # # loop through all tags
    # for previousTagNum in range (len(tagList)):
    #     previousTag = tagList[previousTagNum]
    #     # find total number of occurrences of tag pairs with previousTag
    #     totalPreviousTag = sum(tagPairOccurrenceDict[previousTag].values())
    #     # find count of tags that follow previous tag
    #     tagsFollowing = len(tagPairOccurrenceDict[previousTag])
    #     # loop through all tags again, to find all possible combination of tags
    #     for nextTagNum in range (len(tagList)):
    #         nextTag = tagList[nextTagNum]
    #         # find probabilities of actual tag pairs that have appeared in the training data set
    #         if previousTag in tagPairOccurrenceDict.keys() and nextTag in tagPairOccurrenceDict[previousTag]:
    #             # create the tag pair (nextTag, previousTag)
    #             tagPair = (nextTag, previousTag)
    #             transitionProbabilities[tagPair] = math.log((tagPairOccurrenceDict[previousTag].get(nextTag, 0) + smoothingConstant) / (totalPreviousTag + smoothingConstant*(tagsFollowing + 1)))
    #         # find probabilities of tag pairs that have NOT appeared in the training data set (previousTag could be in tagPairOccurrence as a key, but the next tag might not be paired with it)
    #         elif (previousTag not in tagPairOccurrenceDict.keys() or nextTag not in tagPairOccurrenceDict[previousTag]) and previousTag != 'END':  
    #             transitionProbabilities[(nextTag, previousTag)] = math.log(smoothingConstant / (sum(tagPairOccurrenceDict[previousTag].values()) + smoothingConstant*(len(tagPairOccurrenceDict[previousTag]) + 1)))
    #         elif (previousTag not in tagPairOccurrenceDict.keys() or nextTag not in tagPairOccurrenceDict[previousTag]) and previousTag == 'END':
    #             transitionProbabilities[(nextTag, previousTag)] = math.log(smoothingConstant / smoothingConstant)

    # # # Step 2.2 & 3.2 Compute Transition Probabilities P(tagB|tagA)
    # # # declare a dictionary to store transition probabilities of each tag pair; (tagB, tagA) where tagB follows tagA
    # # transitionProbabilities = {}
    # # # loop through all previous tags
    # # for previousTag in tagPairOccurrenceDict.keys():
    # #     # for each previous tag, loop through all tags that could follow
    # #     for nextTag in tagPairOccurrenceDict[previousTag]:
    # #         # create the tag pair (nextTag, previousTag)
    # #         tagPair = (nextTag, previousTag)
    # #         transitionProbabilities[tagPair] = math.log((tagPairOccurrenceDict[previousTag].get(nextTag, 0) + smoothingConstant) / (sum(tagPairOccurrenceDict[previousTag].values()) + smoothingConstant*(len(tagPairOccurrenceDict[previousTag]) + 1)))
    # #         # ORIGINAL INEFFICIENT METHOD
    # #         # transitionProbabilities[tagPair] = math.log((tagPairOccurrenceDict[tagPair] + smoothingConstant) / (sum(value for key, value in tagPairOccurrenceDict.items() if key[1] == tagPair[1]) + smoothingConstant*(sum(1 for key in tagPairOccurrenceDict if key[1] == tagPair[1]) + 1)))
    
    # # # loop through all tags and if each tag pair doesn't exist, then initialize it
    # # for previousTagNum in range(len(tagList)):
    # #     for nextTagNum in range(len(tagList)):
    # #         if (tagList[nextTagNum], tagList[previousTagNum]) not in transitionProbabilities and tagList[previousTagNum] != 'END':
    # #             transitionProbabilities[(tagList[nextTagNum], tagList[previousTagNum])] = math.log(smoothingConstant / (sum(tagPairOccurrenceDict[tagList[previousTagNum]].values()) + smoothingConstant*(len(tagPairOccurrenceDict[tagList[previousTagNum]]) + 1)))
    # #         elif (tagList[nextTagNum], tagList[previousTagNum]) not in transitionProbabilities and tagList[previousTagNum] == 'END':
    # #             transitionProbabilities[(tagList[nextTagNum], tagList[previousTagNum])] = math.log(smoothingConstant / smoothingConstant)

    # # COMBINE INTO A SINGLE DOUBLE FOR LOOP (LOOP THROUGH ALL WORDS THEN THROUGH ALL TAGS); check if the wordTag pair exists
    # # Step 2.3 & 3.3 Compute Emission Probabilities P(word|tag)
    # # declare a dictionary to store emission probabilities 
    # emissionProbabilities = {}
    # # loop through all tags
    # for tagNum in range (len(tagList)):
    #     # loop through all words
    #     for wordNum in range (len(wordList)):
    #         # find probabilities of actual word, tag pairs that have appeared in the training data set
    #         if tagList[tagNum] in wordTagPairOccurrenceDict.keys() and wordList[wordNum] in wordTagPairOccurrenceDict[tagList[tagNum]]:
    #             # create the word tag pair (word, tag)
    #             wordTagPair = (wordList[wordNum], tagList[tagNum])
    #             emissionProbabilities[wordTagPair] = math.log((wordTagPairOccurrenceDict[tagList[tagNum]].get(wordList[wordNum], 0) + smoothingConstant) / (sum(wordTagPairOccurrenceDict[tagList[tagNum]].values()) + smoothingConstant*(len(wordTagPairOccurrenceDict[tagList[tagNum]]) + 1)))
    #         # find probabilities of word tag pairs that haven't appeared, but the words and tags we have seen in the dat set
    #         elif tagList[tagNum] not in wordTagPairOccurrenceDict.keys() or wordList[wordNum] not in wordTagPairOccurrenceDict[tagList[tagNum]]:
    #             emissionProbabilities[(wordList[wordNum], tagList[tagNum])] = math.log(smoothingConstant / (sum(wordTagPairOccurrenceDict[tagList[tagNum]].values()) + smoothingConstant*(len(wordTagPairOccurrenceDict[tagList[tagNum]]) + 1)))
    #     # add an ('UNKNOWN', tag) for words that were not seen in the training data for each tag 
    #     unknownWordTagPair = ('UNKNOWN', tagList[tagNum])
    #     emissionProbabilities[unknownWordTagPair] = math.log(smoothingConstant / (sum(wordTagPairOccurrenceDict[tagList[tagNum]].values()) + smoothingConstant*(len(wordTagPairOccurrenceDict[tagList[tagNum]]) + 1)))

    # # # Step 2.3 & 3.3 Compute Emission Probabilities P(word|tag)
    # # # declare a dictionary to store emission probabilities 
    # # emissionProbabilities = {}
    # # # loop through all tags
    # # for tag in wordTagPairOccurrenceDict.keys():
    # #     # loop through all words associated with the tag
    # #     for word in wordTagPairOccurrenceDict[tag]:
    # #         # create the word tag pair (word, tag)
    # #         wordTagPair = (word, tag)
    # #         emissionProbabilities[wordTagPair] = math.log((wordTagPairOccurrenceDict[tag].get(word, 0) + smoothingConstant) / (sum(wordTagPairOccurrenceDict[tag].values()) + smoothingConstant*(len(wordTagPairOccurrenceDict[tag]) + 1)))
    # #         # ORIGINAL INEFFICIENET METHOD
    # #         # emissionProbabilities[wordTagPair] = math.log((wordTagPairOccurrenceDict[wordTagPair] + smoothingConstant) / (sum(value for key, value in wordTagPairOccurrenceDict.items() if key[1] == wordTagPair[1]) + smoothingConstant*(sum(1 for key in wordTagPairOccurrenceDict if key[1] == wordTagPair[1]) + 1)))
    # #     # add an ('UNKNOWN', tag) for words that were not seen in the training data for each tag 
    # #     unknownWordTagPair = ('UNKNOWN', tag)
    # #     emissionProbabilities[unknownWordTagPair] = math.log(smoothingConstant / (sum(wordTagPairOccurrenceDict[tag].values()) + smoothingConstant*(len(wordTagPairOccurrenceDict[tag]) + 1)))

    # # # loop through all words and tags and if the word tag pair doesn't exist, then initialize it
    # # for wordNum in range(len(wordList)):
    # #     for tagNum in range(len(tagList)):
    # #         if (wordList[wordNum], tagList[tagNum]) not in wordTagPairList:
    # #             emissionProbabilities[(wordList[wordNum], tagList[tagNum])] = math.log(smoothingConstant / (sum(wordTagPairOccurrenceDict[tagList[tagNum]].values()) + smoothingConstant*(len(wordTagPairOccurrenceDict[tagList[tagNum]]) + 1)))

    # # sanity check
    # # print(initialProbabilities)
    # # print(transitionProbabilities)
    # # print(emissionProbabilities)

    # # Step 4: Construct the Trellis (row dimension = total # of all tags, column dimension = total # of words in a sentence)    AND     Step 5: Return Best Path Through Trellis
    # # declare a 2D list for storing the sentences in the test data with tags
    # sentenceList = [[0.0] for sentenceNum in range (len(test))]
    # # loop through each sentence index in the test data set and construct a 2D list for the trellis for each sentence and also a backpointer 2D list to store pointer to previous tag/time pair in the path
    # for sentenceNum in range (len(test)):
    #     trellis = [[0.0]*len(test[sentenceNum]) for row in range(len(tagList))]
    #     backPointers = [[0.0]*len(test[sentenceNum]) for row in range(len(tagList))]

    #     # initialization step: loop through the entire first column and calculate prior probability and emission probability
    #     for tagNum in range (len(tagList)):
    #         if tagList[tagNum] == 'START':
    #             trellis[tagNum][1] = initialProbabilities[tagList[tagNum]]*emissionProbabilities[(test[sentenceNum][0], tagList[tagNum])]
    #         else:
    #             trellis[tagNum][1] = initialProbabilities[tagList[tagNum]]*emissionProbabilities[('UNKNOWN', tagList[tagNum])]
    #         backPointers[tagNum][1] = 0
        
    #     # recursion step: loop through each time step (or word in the sentence; starting from 2nd word)
    #     for wordNum in range (len(test[sentenceNum]) - 1):
    #         # loop through each state/tag 
    #         for tagNum in range (len(tagList)):
    #             # check if the word is in the wordList, if so, proceed as normal
    #             if test[sentenceNum][wordNum] in wordList:
    #                 # find the maximum probability (from current word to next word) and the previous state 
    #                 maximumProbability, previousState = max((trellis[previousTagNum][wordNum]*transitionProbabilities[(tagList[tagNum], tagList[previousTagNum])]*emissionProbabilities[(test[sentenceNum][wordNum], tagList[tagNum])], previousTagNum) for previousTagNum in range(len(tagList)))
    #             else:
    #                 maximumProbability, previousState = max((trellis[previousTagNum][wordNum]*transitionProbabilities[(tagList[tagNum], tagList[previousTagNum])]*emissionProbabilities[('UNKNOWN', tagList[tagNum])], previousTagNum) for previousTagNum in range(len(tagList)))
    #             # store maximum probability into the trellis and previousState (tag number of previous state) into the backpointer list
    #             trellis[tagNum][wordNum + 1] = maximumProbability
    #             backPointers[tagNum][wordNum + 1] = previousState

    #     # termination step: find the best path probability and the best path pointer (tagNum of the final tag)
    #     bestpathprob, bestpathpointer = max((trellis[tagNum][len(test[sentenceNum])], tagNum) for tagNum in range (len(tagList)))

    #     # backtracking step: starting from the final word, go back to find the previous best tag, inserting it to the front
    #     for wordNum in range (len(test[sentenceNum]) - 1, -1, -1):
    #         # insert the best tag for the current word (starting from the last word)
    #         sentenceList[sentenceNum].insert(0, (test[sentenceNum][wordNum], tagList[bestpathpointer]))
    #         bestpathpointer = backPointers[bestpathpointer][wordNum]

    # return sentenceList

def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    raise NotImplementedError("You need to write this part!")



