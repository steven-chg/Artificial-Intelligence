'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    # raise RuntimeError("You need to write this part!")'

    # Declare an array for tracking number of times word0 appears in each file
    numCountArray = []
    # numCountArray = np.array(numCountArray)
    # Declare an array for tracking P(X0 = x0)
    Pmarginal = []
    # Pmarginal = np.array(Pmarginal)

    # Go through all the documents and track how many times word0 appears in each document using an array of length len(texts) or number of documents
    numDocuments = len(texts)
    for i in range (numDocuments):
        # find number of times word0 occurs in document i and insert it into the numCountArray 
        numCount = texts[i].count(word0)
        # numCountArray = np.append(numCountArray, [numCount])
        numCountArray.append(numCount)

    # Find the largest number in the newly created array and loop for largest number + 1 times (+1 because we also want to include probability of 0 instances)
    largestX0 = max(numCountArray)
    for j in range (largestX0+1):
        # In each loop, count how many documents contain j instances of word0 then divide by total # of documents then insert to Pmarginal array
        numDocumentsWithx0 = numCountArray.count(j)
        probX0 = numDocumentsWithx0/numDocuments
        # Pmarginal = np.append(Pmarginal, [probX0])
        Pmarginal.append(numDocumentsWithx0/numDocuments)

    # Make the list into a numpy array
    Pmarginal = np.array(Pmarginal)
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    # raise RuntimeError("You need to write this part!")

    # Declare arrays for tracking number of times word0 and word1 appears in each file
    numCountArrayWord0 = []
    numCountArrayWord1 = []

    # Go through all the documents and track how many times word0 and word1 appears in each document using an array of length len(texts) or number of documents
    numDocuments = len(texts)
    for i in range (numDocuments):
        # find number of times word0 occurs in document i and insert it into the numCountArrayWord0
        numCountWord0 = texts[i].count(word0)
        numCountArrayWord0.append(numCountWord0)
        # find number of times word1 occurs in document i and insert it into the numCountArrayWord1
        numCountWord1 = texts[i].count(word1)
        numCountArrayWord1.append(numCountWord1)

    # Set up empty 2D array Pcond with shape cX0 (row), cX1 (column); initialize to 0 (+1 to also include count of 0)
    Pcond = [[0]*(max(numCountArrayWord1)+1) for i in range(max(numCountArrayWord0)+1)]

    # Loop through every row of Pcond
    for j in range (max(numCountArrayWord0)+1):
        # Loop through every column of Pcond
        for z in range (max(numCountArrayWord1)+1):
            # Find how many documents have j instances of word0 and then divide by total number of documents for P(X0 = j)
            probX0 = numCountArrayWord0.count(j)/numDocuments
            
            # Calculate joint probability P(X0 = j, X1 = z)
            numDocsWithx0x1 = 0
            # Go through every document and see if it has j word0 and z word1, if so, increment numDocsWithx0x1 counter
            for a in range (numDocuments):
                if numCountArrayWord0[a] == j and numCountArrayWord1[a] == z:
                  numDocsWithx0x1+=1
            probX0X1 = numDocsWithx0x1/numDocuments


            # Find the conditional probability and insert it into Pcond
            if probX0 != 0:
              probX1GivenX0 = probX0X1/probX0
            else: 
              probX1GivenX0 = np.nan
            
            Pcond[j][z] = probX1GivenX0

    # Make the 2D array into a numpy array
    Pcond = np.array(Pcond)
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    # raise RuntimeError("You need to write this part!")

    # Get the maximum number of instances of each word in the documents (subtract 1 to account for 0 instances)
    maximumNumWord0 = np.shape(Pcond)[0] - 1
    maximumNumWord1 = np.shape(Pcond)[1] - 1

    # Declare Pjoint array and initialize to 0
    Pjoint = [[0]*(maximumNumWord1+1) for i in range(maximumNumWord0+1)]

    # Loop through the entire Pjoint 2D array
    for i in range (maximumNumWord0 + 1):
        for j in range (maximumNumWord1 + 1):
            # Find the joint probability (if marginal probability is 0, then joint probability is 0)
            if Pmarginal[i] != 0:
                jointProbability = Pmarginal[i]*Pcond[i][j]
            else:
                jointProbability = 0
            # Insert into 2D array
            Pjoint[i][j] = jointProbability

    # Make the 2D array into a numpy array
    Pjoint = np.array(Pjoint)
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    # raise RuntimeError("You need to write this part!")

    # Declare empty 2D array and variables to keep track of mean of X0 and X1
    mu = []
    meanofX0 = 0
    meanofX1 = 0

    # Find the mean of X0 and X1; Loop through each column in each row
    for i in range (np.shape(Pjoint)[0]):
      for j in range (np.shape(Pjoint)[1]):
        # Multiply the current x0 and x1 by the probability 
        meanofX0 += i*Pjoint[i][j]
        meanofX1 += j*Pjoint[i][j]

    # Insert the means into the array
    mu.append(meanofX0)
    mu.append(meanofX1)

    # Make the 2D array into a numpy array
    mu = np.array(mu)
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    # raise RuntimeError("You need to write this part!")

    # Declare Sigma array and initialize to 0
    Sigma = [[0]*2 for i in range(2)]

    # Declare variables for the variances of X0, X1, and covariance of X0, X1
    varianceX0 = 0
    varianceX1 = 0
    covarianceX0X1 = 0

    # Loop through the entire Pjoint numpy array to calculate variances and covariance
    for i in range (np.shape(Pjoint)[0]):
        for j in range (np.shape(Pjoint)[1]):
          varianceX0 += (i-mu[0])*(i-mu[0])*Pjoint[i][j]
          varianceX1 += (j-mu[1])*(j-mu[1])*Pjoint[i][j]
          covarianceX0X1 += (i-mu[0])*(j-mu[1])*Pjoint[i][j]
            
    # Insert results into array and turn into numpy array
    Sigma[0][0] = varianceX0
    Sigma[0][1] = covarianceX0X1
    Sigma[1][0] = covarianceX0X1
    Sigma[1][1] = varianceX1
    Sigma = np.array(Sigma)

    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    # raise RuntimeError("You need to write this part!")
  
    # Create a new empty counter
    Pfunc = Counter()

    # Loop through entire Pjoint array
    for i in range (np.shape(Pjoint)[0]):
       for j in range (np.shape(Pjoint)[1]):
          # Retrieve the function output, then set the hash key as the output and its value as Pjoint[i][j] 
          hashKey = f(i, j)
          hashValue = Pjoint[i][j]
          # Update the value/probability 
          Pfunc[hashKey] += hashValue
          
    return Pfunc
    
