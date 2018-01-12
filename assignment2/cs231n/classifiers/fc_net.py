from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: [get]Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        '''
        layer1: (N,D)--->affine(D,H)--->(N,H)--->ReLU
        layer2: (N,H)--->affine(H,C)--->(N,C)--->scores--->softmax
        '''
        D, H, C =input_dim, hidden_dim, num_classes
        self.params['W1'] = np.random.randn(D, H) * weight_scale
        self.params['b1'] = np.zeros(H)
        self.params['W2'] = np.random.randn(H, C) * weight_scale
        self.params['b2'] = np.zeros(C)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: [get]Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        out1, cache1 = affine_relu_forward(X, W1, b1)
        out2, cache2 = affine_forward(out1, W2, b2)
        #scores
        scores = out2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: [get]Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        dx2, grads['W2'], grads['b2'] = affine_backward(dx, cache2)
        dx1, grads['W1'], grads['b1'] = affine_relu_backward(dx2, cache1)
        loss += 0.5 * self.reg * (np.sum(W1 **2) + np.sum(W2 ** 2))
        #regulation
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: [get]Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        '''
        first layer: (N, D)--->bn--->relu--->dropout--->(N, D)*(D, H[0]) = (N, H[0])
        second layer: (N, H[0])--->(N, H[1])
        ...
        ouput layer: (N, H[-1])--->(N, C)
        L = len(H) + 1  H: 0~L-2#  -1 == L - 2
        '''
        D, C, L = input_dim, num_classes, self.num_layers
        #L layers in total, first layer, ouput layer, L-2 other hidden layers
        #first layer
        self.params['W1'] = np.random.randn(D, hidden_dims[0]) * weight_scale#(D,H)
        self.params['b1'] = np.zeros(hidden_dims[0])
        #hidden layers
        #%map is also available
        for i in range(L - 2):
            self.params['W' + str(i + 2)] = np.random.randn(hidden_dims[i], hidden_dims[i + 1]) * weight_scale#(H,H)
            self.params['b' + str(i + 2)] = np.zeros(hidden_dims[i + 1])#(H,)
            if use_batchnorm:
                self.params['gamma' + str(i + 1)] = np.ones(hidden_dims[i])
                self.params['beta' + str(i + 1)] = np.zeros(hidden_dims[i])
        #output layer
        self.params['W' + str(L)] = np.random.randn(hidden_dims[-1], C) * weight_scale#(H,C)
        self.params['b' + str(L)] = np.zeros(C, dtype = self.dtype)#(C,)
        if use_batchnorm:
            self.params['gamma' + str(L - 1)] = np.ones(hidden_dims[-1])
            self.params['beta' + str(L - 1)] = np.zeros(hidden_dims[-1])
        ###others' code. it's nice#############################################################
        # layer_input_dim = input_dim
        # for i, hd in enumerate(hidden_dims):
        #     self.params['W%d' % (i + 1)] = weight_scale * np.random.randn(layer_input_dim, hd)
        #     self.params['b%d' % (i + 1)] = weight_scale * np.zeros(hd)
        #     if self.use_batchnorm:
        #         self.params['gamma%d' % (i + 1)] = np.ones(hd)
        #         self.params['beta%d' % (i + 1)] = np.zeros(hd)
        #     layer_input_dim = hd
        # self.params['W%d' % (self.num_layers)] = weight_scale * np.random.randn(layer_input_dim, num_classes)
        # self.params['b%d' % (self.num_layers)] = weight_scale * np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: [get but not check]Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        '''''''{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax'''
        # affine_out, bn_out, relu_out, drop_out = {}, {}, {}, {}
        # affine_cache, bn_cache, relu_cache, drop_cache = {}, {}, {}, {}
        # out, cache = {}, {}
        # #hidden layers
        # L = self.num_layers
        # out[0] = X
        # for i in range(L - 1):
        #     '''affine'''
        #     affine_out[i + 1], affine_cache[i + 1] = affine_forward(out[i], self.params['W' + str(i + 1)], self.params['b' + str(i + 1)])
        #     '''batchnorm + relu'''
        #     if self.use_batchnorm:
        #         bn_out[i + 1], bn_cache[i + 1] = batchnorm_forward(affine_out[i + 1], self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)], self.bn_params[i])
        #         relu_out[i + 1], relu_cache[i + 1] = relu_forward(bn_out[i + 1])
        #     #'''only relu'''
        #     else:
        #         relu_out[i + 1], relu_cache[i + 1] = relu_forward(affine_out[i+1])
        #     '''dropout, its cache must be saved in another dict'''
        #     if self.use_dropout:
        #         drop_out[i + 1], drop_cache[i + 1] = dropout_forward(relu_out[i+1], self.dropout_param)
        #         out[i + 1] = drop_out[i + 1]
        #         cache[i + 1] = drop_cache[i + 1]
        #     else:
        #         out[i + 1] = relu_out[i + 1]
        #         cache[i + 1] = relu_cache[i + 1]
        # #output layer
        # out[L], cache[L] = affine_forward(out[L - 1], self.params['W' + str(L)], self.params['b' + str(L)])
        # scores = out[L]
        '''other's code'''
        hidden_layers, caches = {}, {}
        dp_caches = {}
        hidden_layers[0] = X
        W = self.params['W1']
        for i in range(self.num_layers):
            W, b = self.params['W' + str(i+1)], self.params['b' + str(i+1)]
            if i == self.num_layers-1:
                hidden_layers[i+1], caches[i] = affine_forward(hidden_layers[i], W, b)
            else:
                if self.use_batchnorm:
                    gamma, beta = self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)]
                    hidden_layers[i+1], caches[i] = affine_bn_relu_forward(hidden_layers[i], W, b, gamma, beta, self.bn_params[i])
                else:
                    hidden_layers[i+1], caches[i] = affine_relu_forward(hidden_layers[i], W, b)
                if self.use_dropout:
                    hidden_layers[i+1], dp_caches[i] = dropout_forward(hidden_layers[i+1], self.dropout_param)

        scores = hidden_layers[self.num_layers]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: [get but not check]Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # dx = {}
        # loss, dx[L] = softmax_loss(scores, y)
        # #output layer
        # dx[L - 1], grads['W' + str(L)], grads['b' + str(L)] = affine_backward(dx[L], cache[L])
        # #L-1 hidden layers
        # for i in range(L - 1):
        #     k = L - 2 - i
        #     #dropout
        #     if self.use_dropout:#dx_ is the temporary dx
        #         dx_temp1 = dropout_backward(dx[k + 1], drop_cache[k])
        #     else:
        #         dx_temp1 = dx[k + 1]
        #     #relu
        #     dx_temp2 = relu_backward(dx_temp1, relu_cache[k])
        #     #bn
        #     if self.use_batchnorm:
        #         dx_temp3, dgamma, dbeta = batchnorm_backward(dx_temp2, bn_cache[k])
        #     else:
        #         dx_temp3 = dx_temp2
        #     #affine
        #     dx[k], grads['W' + str(k)], grads['b' + str(k)] = affine_backward(dx_temp3, affine_cache[k])
        #     #regulation
        #     loss += self.reg * np.sum(self.params['W' + str(k)] ** 2) * 0.5
        #     grads['W' + str(k)] += self.reg * self.params['W' + str(k)]
        '''
        other's code
        '''
        loss, dscore = softmax_loss(scores, y)
        dhiddens = {}
        dhiddens[self.num_layers] = dscore
        for i in range(self.num_layers, 0, -1):
            if i == self.num_layers:
                dhiddens[i-1], grads['W'+str(i)], grads['b'+str(i)] = \
                    affine_backward(dhiddens[i], caches[i-1])
            else:
                if self.use_dropout:
                    dhiddens[i] = dropout_backward(dhiddens[i], dp_caches[i-1])
                if self.use_batchnorm:
                    dhiddens[i-1], grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = \
                        affine_bn_relu_backward(dhiddens[i], caches[i-1])
                else:
                    dhiddens[i-1], grads['W' + str(i)], grads['b' + str(i)] = \
                        affine_relu_backward(dhiddens[i], caches[i - 1])
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i)] ** 2)
            grads['W'+str(i)] += self.reg * self.params['W'+str(i)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
