'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter
import math 

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    # raise RuntimeError("You need to write this part!")

    # declare dictionary of counters 
    frequency = {'pos': Counter(), 'neg': Counter()}

    # loop through each class ('pos' and 'neg'); i takes on the value 'pos' and 'neg' instead of integers 
    for classType in train:
        # loop through each text (0 through 1999 for 'neg' and 0 through 5999 for 'pos')
        for text in range (len(train[classType])):
            # loop through index of each token/word until the index of 2nd to last token/word (since we are looking at bigrams)
            for wordIndex in range (len(train[classType][text]) - 1):
                # get the current word and the next word to form the bigarm 
                bigram = train[classType][text][wordIndex]+"*-*-*-*"+train[classType][text][wordIndex+1]
                # update the counter in the dictionary 
                frequency[classType].update({bigram: 1})

    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    # raise RuntimeError("You need to write this part!")

    # create new dictionary of counters to return
    nonstop = {'pos': Counter(), 'neg': Counter()}

    # loop through each class ('pos' and 'neg')
    for classType in frequency:
        # loop through each bigram in each class (bigram is the bigram string)
        for bigram in frequency[classType]:
            # split the bigram string
            bigramSplit = bigram.split('*-*-*-*')
            # check if either word is not a stopword
            if (bigramSplit[0] not in stopwords) or (bigramSplit[1] not in stopwords):
                # if so, add the bigram into the new dictionary and 'import' its count from the old dictionary
                nonstop[classType].update({bigram: frequency[classType][bigram]})

    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    # raise RuntimeError("You need to write this part!")

    # declare likelihood dictionary 
    likelihood = {'pos': {}, 'neg': {}}

    # loop through each class ('pos' and 'neg')
    for classType in nonstop:
        # get the number of bigram types for each class type
        numBigramTypes = len(nonstop[classType])

        # get the total # of tokens of any bigram in texts for each class type (sum all the counters within each class type)
        totalBigrams = sum(nonstop[classType][bigram] for bigram in nonstop[classType])
 
        # insert the likelihood of an out-of-vocabulary bigram for each class type
        likelihood[classType]["OOV"] = smoothness/(totalBigrams + smoothness*(numBigramTypes + 1))

        # loop through each bigram in each class
        for bigram in nonstop[classType]:
            likelihood[classType][bigram] = (nonstop[classType][bigram] + smoothness)/(totalBigrams + smoothness*(numBigramTypes + 1))

    return likelihood         


def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    # raise RuntimeError("You need to write this part!")

    # declare empty list for hypotheses 
    hypotheses = []

    # loop through index of each text/document
    for text in range (len(texts)):
        # for each text, start calculating P(Class = pos|Text = x) and P(Class = neg|Text = x); take the log 
        probClassPos = math.log(prior)
        probClassNeg = math.log(1-prior)

        # loop through the index of every token/word in the text until the 2nd to last index
        for wordIndex in range (len(texts[text]) - 1):
            # form the bigram 
            bigram = texts[text][wordIndex] + "*-*-*-*" + texts[text][wordIndex + 1]

            # check for stop bigrams
            bigramSplit = bigram.split("*-*-*-*")
            if bigramSplit[0] not in stopwords or bigramSplit[1] not in stopwords:
                # check and see if the bigram exists in likelihood; if not, then find it using "OOV" index; otherwise, use the bigram as index
                if bigram not in likelihood['pos'] and bigram not in likelihood['neg']:
                    probClassPos += math.log(likelihood['pos']["OOV"])
                    probClassNeg += math.log(likelihood['neg']["OOV"])
                elif bigram not in likelihood['pos'] and bigram in likelihood['neg']:
                    probClassPos += math.log(likelihood['pos']["OOV"])
                    probClassNeg += math.log(likelihood['neg'][bigram])
                elif bigram in likelihood['pos'] and bigram not in likelihood['neg']:
                    probClassPos += math.log(likelihood['pos'][bigram])
                    probClassNeg += math.log(likelihood['neg']["OOV"])
                else:
                    probClassPos += math.log(likelihood['pos'][bigram])
                    probClassNeg += math.log(likelihood['neg'][bigram])

        # After finding P(Class = pos|Text = x) and P(Class = neg|Text = x), estimate the class and append into the list
        if probClassPos > probClassNeg:
            hypotheses.append("pos")
        elif probClassPos < probClassNeg:
            hypotheses.append("neg")
        else:
            hypotheses.append("undecided")

    return hypotheses 


def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    # raise RuntimeError("You need to write this part!")

    # declare 2D accuracies array and initialize to 0
    accuracies = [[0]*len(smoothnesses) for i in range(len(priors))]

    # loop through every row/prior value
    for priorIndex in range (len(priors)):
        # loop through every column/smoothness value
        for smoothnessIndex in range (len(smoothnesses)):
            # declare and initialize variable that keeps track of number of correct class estimations to 0
            correctEstimate = 0
        
            # get the likelihood matrix for the current smoothness
            likelihood = laplace_smoothing(nonstop, smoothnesses[smoothnessIndex])

            # get the hypotheses for every single text document
            hypotheses = naive_bayes(texts, likelihood, priors[priorIndex])

            # loop through all the hypotheses and check if it is correct 
            for hypothesisIndex in range (len(hypotheses)):
                # check if estimate is correct
                if hypotheses[hypothesisIndex] == labels[hypothesisIndex]:
                    correctEstimate += 1

            # find and insert the accuracy
            accuracies[priorIndex][smoothnessIndex] = correctEstimate / len(labels)

    # convert 2D array into 2D numpy array
    accuracies = np.array(accuracies)
    return accuracies

                          