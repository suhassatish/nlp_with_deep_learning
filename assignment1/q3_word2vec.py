#!/usr/bin/env python
"""
In this module, the underlying implementation for word2vec uses 2 models - either skipGram or continuous bag of words
(CBOW).

Each of those 2 models can use one of 2 cost functions - either softmax or negative-sampling, each with their own
corresponding derivatives (aka gradients)
"""
import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    # this is the same as norm function which takes sqrt(sum of squares of elements along each matrix-row)
    norm_row = np.sqrt(np.sum(x**2, axis=1))  # axis = 0 is column-wise sum. axis=1 is row-wise sum
    # returns [5, 2.25] for below example in `test_normalize_rows`

    norm_col = np.reshape(norm_row, (-1, 1))  # reshape with as many `rows` as the length of the input array (-1) and
    # 1 column
    # gives [[5], [2.25]] or in other words, matrix transpose
    x /= norm_col
    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(vc_predicted, target, uw_output, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    vc_predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    uw_output -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    dJ_vc -- the gradient with respect to the predicted word
           vector
    dJ_uw -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    N = uw_output.shape[0]  # n_words: vocab size
    y = np.zeros(N)
    y[target] = 1  # y is a 1-hot encoded vector with the actual word's index being 1 and rest of elements being 0

    score = np.dot(vc_predicted, uw_output.T)  # vc dot uo_transpose which gives a vector of dimension (1, n_words)
    y_hat = softmax(score)

    # cross-entropy cost is given by formula in assignment 1.2b
    cost = np.sum(-y * np.log(y_hat))

    dout = y_hat - y  # (1, n_words)

    grad_pred_dJ_vc = np.dot(dout, uw_output)  # (1, dim_embed)

    grad_dJ_uw = np.dot(dout.T, vc_predicted)  # (n_words, dim_embed)

    return cost, grad_pred_dJ_vc, grad_dJ_uw


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted_vc, target, outputVectors_uk, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    cost = 0.0
    sigmd_uoT_vc = sigmoid(np.dot(predicted_vc.reshape(-1), outputVectors_uk[target].T))
    cost += -np.log(sigmd_uoT_vc)

    gradPred_dJ_vc = np.zeros_like(predicted_vc)
    gradPred_dJ_vc += (sigmd_uoT_vc - 1) * outputVectors_uk[target]

    grad_dJ_uw = np.zeros_like(outputVectors_uk)
    grad_dJ_uw[target:target + 1] = (sigmd_uoT_vc - 1) * predicted_vc

    neg_samples = []
    for i in range(K):
        j = dataset.sampleTokenIdx()
        if j == target or (j in neg_samples):
            i -= 1  # if negative sample is same with target or already sampled, then resample.
            continue
        neg_samples.append(j)

        sigmd_ukT_vc = sigmoid(-np.dot(predicted_vc.reshape(-1), outputVectors_uk[j].T))
        cost += -np.log(sigmd_ukT_vc)  # cost for negative sample

        grad_dJ_uw[j:j + 1] = (1 - sigmd_ukT_vc) * predicted_vc  # gradient for negative sample
        gradPred_dJ_vc += (1 - sigmd_ukT_vc) * outputVectors_uk[j]

    return cost, gradPred_dJ_vc, grad_dJ_uw


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    idx = tokens[currentWord]  # tokens['a'] = 1
    input_vector = inputVectors[idx:idx+1]

    for context in contextWords:
        c, g_in, g_out = word2vecCostAndGradient(input_vector, tokens[currentWord], outputVectors, dataset)
        cost += c
        gradIn[idx:idx+1, :] += g_in
        gradOut += g_out

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    for contextWord in contextWords:
        idx = tokens[contextWord]  # tokens['a'] = 1
        input_vector = inputVectors[idx:idx + 1]
        c, g_in, g_out = word2vecCostAndGradient(input_vector, tokens[currentWord], outputVectors, dataset)
        cost += c
        gradIn[idx:idx + 1, :] += g_in
        gradOut += g_out

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
