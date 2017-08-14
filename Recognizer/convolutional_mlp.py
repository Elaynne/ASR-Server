
import os
import sys
import time
import dill
import numpy as np
import dill
import pickle
import theano
import datetime
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_hmm import LogisticRegression, load_data
from mlp import HiddenLayer
import utils

# area_rotulada = np.array(["ai","am","am","am","am","af","di","dm","dm","dm","dm","dm","dm","df","ei","em","em","em","em","em","em","ef", "pi","pm","pm","pm","pm","pf", "ri","rm","rm","rm","rm","rm","rf"])




class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]



def evaluate_lenet5(learning_rate=0.1, n_epochs=100,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=100):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = np.random.RandomState(23455)

    largura = 70
    rodada = 10
    datasets,xTemp = load_data(dataset, rodada, largura= largura)
    #Recupera ParÂmetros
    _,_,qt_rotulos = utils.leitura_arquivo(largura)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print(test_set_x.shape.eval())
    print(train_set_x.shape.eval())
    tes1 = xTemp  # theano.shared(xTemp, borrow=True)
    tes2 = xTemp  # theano.shared(xTemp, borrow=True)
    tes3 = xTemp  # theano.shared(xTemp, borrow=True)
    tes4 = xTemp  # theano.shared(xTemp, borrow=True)


    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    ntes1 = tes1.get_value(borrow=True).shape[0]
    ntes2 = tes2.get_value(borrow=True).shape[0]
    ntes3 = tes3.get_value(borrow=True).shape[0]
    ntes4 = tes4.get_value(borrow=True).shape[0]

    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    ntes1 /= batch_size
    ntes2 /= batch_size
    ntes3 /= batch_size
    ntes4 /= batch_size
    print(n_train_batches,n_valid_batches,n_test_batches)
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    base = T.matrix('base')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print( '... building the model')
    print(datetime.datetime.now())

                                                            # 60 - 3 + 1 = 58   MaxPool: 58/2  = 29
    layer0_input = x.reshape((batch_size, 1, largura, 40))  # (30-3+1   40-3+1 = (28,38)   MaxPooling:  (28/2   , 38/2) = (14, 19)

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,            # ( 35 - 3 +1 , 40 -3 +1 )  = ( 32 , 38) : Max ( 32/2  38/2) = ( 16, 19)
        image_shape=(batch_size, 1, largura, 40), # ( 25-3+1   38) = (23 , 38 )  MaxPo :  (23/2  38/2) = (  11 ,19)
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )

    img_shape1 = int(np.trunc((largura - 3 +1) / 2))
    print(img_shape1)
    layer1 = LeNetConvPoolLayer(
        rng,                                               # ( 24 - 3 + 1 ) = ( 22 )   = ( 22/2 ) = 11
        input=layer0.output,                            # ( 16 -3+1 = 14)   ( 14/2 = 7)
        image_shape=(batch_size, nkerns[0], img_shape1, 19),  # ( 14 -3 +1  , 19-3+1 ) = ( 12 , 17 )
        filter_shape=(nkerns[1], nkerns[0], 3, 3),    # (12 /2 , 17/2 )  = (6, 8)
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 20300) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    l2_out = 250
    img_shape2 = int(np.trunc((img_shape1 -3 + 1)/2))

    print(img_shape2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * img_shape2 * 8,
        n_out=l2_out,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=l2_out, n_out=qt_rotulos)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    model_predict1 = theano.function([index], layer3.y_pred,
         givens={\
            x: tes1[index * batch_size: (index + 1) * batch_size]})
    model_predict2 = theano.function([index], layer3.y_pred,
         givens={\
            x: tes2[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params+ layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 50000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    #Salvation
    layer3.salvation = x

    print("LEEEEEEN", len(train_set_x.eval()))
    while (epoch < n_epochs):

        print(epoch)
        epoch = epoch + 1

        lista = np.arange(0,len(train_set_x.eval()))
        np.random.shuffle(lista)
        #
        train_set_x = train_set_x[lista]
        train_set_y = train_set_y[lista]

        #print('épocas',epoch)
        for minibatch_index in range(int(n_train_batches)):

            iter = (epoch - 1) * int(n_train_batches) + int(minibatch_index)

            if iter % 100 == 0:
                print('training @ iter = ', iter)

            # print(train_set_y.eval()[minibatch_index * batch_size: (minibatch_index + 1) * batch_size])
            cost_ij = train_model(minibatch_index)
            # print('minibatch_index', minibatch_index)
            #print('custo',cost_ij)
            # print('wow', minibatch_index * batch_size, (minibatch_index + 1) * batch_size)


            if (iter + 1) % validation_frequency == 0:
                print('wow')
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(int(n_valid_batches))]
                this_validation_loss = np.mean(validation_losses)
                print(('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.)))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(int(n_test_batches))
                    ]

                    test_score = np.mean(test_losses)

                    print((('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.)))


            if patience <= iter:
                done_looping = True
                break
        print(('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.)))

    end_time = time.clock()
    print('Optimization complete.')
    print(('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.)))
    print(('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    with open('rede.pkl', 'wb') as f:
        dill.dump(layer3, f)

    print('Modelo foi salvo com sucesso')
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
