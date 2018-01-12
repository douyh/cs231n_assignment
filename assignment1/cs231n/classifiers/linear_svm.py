import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] += -X[i]#对于每一个j而言，当j满足了margin>0的条件之后，再来看这里的情况，margin<=0的时候，不会对loss产生贡献，1 by D
        dW[:, j] += X[i]##对于每个j而言的梯度，是X[i],而对于y[i]的梯度，每次出现margin>0都会产生一次累加效果，1 by D

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * np.sum(W * W)
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)  # N by C
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores_correct = scores[np.arange(num_train), y]  # 1 by N
  scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1
  margins = scores - scores_correct + 1.0  # N by C
  margins[np.arange(num_train), y] = 0.0#去掉正确的分类对损失函数的影响，对于SVM来说，只有比正确的分类小1以内的或者更大的才会有影响。
  margins[margins <= 0] = 0.0#去掉这些跟小的负数的影响
  loss += np.sum(margins) / num_train#剩余的都是在gap内的
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margins[margins > 0] = 1.0  # 示性函数的意义
  row_sum = np.sum(margins, axis=1)  # 1 by N
  margins[np.arange(num_train), y] = -row_sum
  dW += np.dot(X.T, margins) / num_train + reg * W  # D by C
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
