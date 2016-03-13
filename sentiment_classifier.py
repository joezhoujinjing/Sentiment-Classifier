import random
import collections
import math
import sys
from collections import Counter
from util import *

############################################################
# Feature extraction
def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    Applying stemming, excluding preps pron etc.
    """
    # -------------------------------------------------------------------------
    words = x.split()
    features = {}
    for word in words:
        if word not in ['the','a','to','of','in','into','is','are','were','was','as','not',',','for','it']:
            if len(word)>6:
                word = word[:-int(0.2*len(word))]
            if features.has_key(word):
                features[word]=features.get(word)+1
            else:
                features[word]=1
    #print features
    return features
    raise Exception("Not implemented yet")
    # -------------------------------------------------------------------------

############################################################
# Stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, 
    @return the weight vector (sparse feature vector) learned.
    '''
    weights = {}  # feature => weight
    # -------------------------------------------------------------------------
    for iteration in range(6):
        for i,dtrain in enumerate(trainExamples):
            features= featureExtractor(dtrain[0])
            y=dtrain[1]
            predictor = dotProduct(weights,features)*y
            if predictor<=1:
                dLoss={}
                for v in features:
                    dLoss[v]=features.get(v)*(-y)
                increment(weights,-1/math.sqrt(i+1),dLoss)
    return weights
    raise Exception("Not implemented yet")
    # -------------------------------------------------------------------------

############################################################
# Generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    #random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # -------------------------------------------------------------------------
        phi = {}
        numInPhi=random.randint(5,20)
        for x in range(numInPhi):
            key = random.choice(weights.keys())
            value = weights.get(key)-random.random()
            phi[key]=value
        if dotProduct(phi,weights)>0:
            y=1
        else: y =0
        # -------------------------------------------------------------------------
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # -------------------------------------------------------------------------
        sentence=""
        sentence=sentence.join(x.split())
        features = {}
        for i in range(len(sentence)-n+1):
            pattern = sentence[i:i+n]
            if features.has_key(pattern):
                features[pattern]=features.get(pattern)+1
            else:
                features[pattern]=1
        return features
        raise Exception("Not implemented yet")
        # -------------------------------------------------------------------------
    return extract

############################################################
# Extra features

def extractExtraCreditFeatures(x):
    # -------------------------------------------------------------------------
    raise Exception("Not implemented yet")
    # -------------------------------------------------------------------------

############################################################
# K-MEANS

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # -------------------------------------------------------------------------
    x=examples
    z=[None]*len(x)
    center=[random.choice(x) for i in range(K)]
    iteration = 0
    totalCost=0
    n=0
    oldCost=0
    while True:
        if n==maxIters:
            break
        n=n+1
        #Step 1, to get the assignment function
        totalCost=0
        for i,point in enumerate(x):
            cost,z[i]=min(
                (math.sqrt(dotProduct(minus(x[i],-1,center[j]),minus(x[i],-1,center[j])))
                          ,j) for j in range(K))
            totalCost=cost+totalCost
        if totalCost-oldCost==0:
            break
        else:
            oldCost=totalCost
        #Step 2, to update center
        for j in range(K):
            center[j]={}
            cluster=[p for i,p in enumerate(x) if z[i]==j]
            magnitude=len(cluster)
            for p in cluster:
                center[j]=minus(center[j],1,p)
            increment(center[j],1.0/magnitude-1,center[j])
    # -------------------------------------------------------------------------
    return center,z,totalCost
    raise Exception("Not implemented yet")
def trueCopy(d1):
    """
    To implement real copy of vectors, not just pointers
    """
    # -------------------------------------------------------------------------
    copy=dict()
    for f,v in d1.items():
        copy[f]=d1.get(f,0)
    # -------------------------------------------------------------------------
    return copy
def minus(d1,scale,d2):
    """
    To implement add and substract, without changing the initial vectors,
    and return a new vector
    """
    # -------------------------------------------------------------------------
    res=trueCopy(d1)
    for f,v in d2.items():
        res[f]=res.get(f,0)+v*scale*1.0
    # -------------------------------------------------------------------------
    return res

if __name__ == '__main__':
    trainExamples = readExamples('polarity.train')
    devExamples = readExamples('polarity.dev')
    featureExtractor = extractWordFeatures
    weights = learnPredictor(trainExamples, devExamples, featureExtractor)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print "Official: train error = %s, dev error = %s" % (trainError, devError)
    

    # basic test for k-means
    x1 = {0:0, 1:0}
    x2 = {0:0, 1:1}
    x3 = {0:0, 1:2}
    x4 = {0:0, 1:3}
    x5 = {0:0, 1:4}
    x6 = {0:0, 1:5}
    examples = [x1, x2, x3, x4, x5, x6]
    centers, assignments, totalCost = kmeans(examples, 2, maxIters=10)
    print 'centers: ',centers
    print 'assignments: ',assignments
    print 'total cost: ',totalCost