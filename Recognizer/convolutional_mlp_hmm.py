"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import dill
import numpy as np
import mfcc
import dill
import scipy.io.wavfile as wave
from pylab import *
import pickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_hmm import LogisticRegression, load_data
from mlp import HiddenLayer

fonemas = np.array(["a","v","an","c","i","p","r","rr","e","k","u","d","e_c","t","s","sil"])

def extrair(arquivo):
    rate = 8000
    sinal = wave.read(arquivo, mmap = 'false')
    m = mfcc.mfcc(200,rate,13)
    return m.extrair_fbank_mel(sinal[1])

def extrair_esp(arquivo):
    rate = 8000
    sinal = wave.read(arquivo, mmap = 'false')
    dt = 0.005
    Fs = int(1.0/dt)
    quadro = 50
    calc_quadro = ((257 * quadro) - len(sinal)) / quadro
    Pxx, freqs, bins, im = specgram(sinal[1], NFFT=256, Fs=Fs, noverlap=quadro,
                                cmap=cm.gist_heat)
    while (Pxx.shape[1] < 70):
        quadro += 1
        Pxx, freqs, bins, im = specgram(sinal[1], NFFT=256, Fs=Fs, noverlap=quadro,
                                cmap=cm.gist_heat)

    return im.get_array()

def classe(arquivo):
    if (arquivo.find('direita') > 0):
        return 'direita'
    if (arquivo.find('esquerda') > 0):
        return 'esquerda'
    if (arquivo.find('recue') > 0):
        return 'recue'
    if (arquivo.find('pare') > 0):
        return 'pare'
    if (arquivo.find('avance') > 0):
        return 'avance'

def classe_id(arquivo):
    if (arquivo.find('direita') > 0):
        return 2
    if (arquivo.find('esquerda') > 0):
        return 3
    if (arquivo.find('recue') > 0):
        return 5
    if (arquivo.find('pare') > 0):
        return 4
    if (arquivo.find('avance') > 0):
        return 1

def seq_frame(input, ind, n):
    x,y = input.shape[0],input.shape[1]
    f = np.zeros((n,y))
    pos = int(n/2)
    f[pos] = input[ind]
    if (ind < pos):
        f[(pos-ind):pos,:] = input[0:ind,:]
        f[pos+1:,:] = input[ind+1:ind+pos+1,:]
    elif ((x-ind-1) < (pos) ):
        f[(pos+1):(pos+x-ind),:] = input[ind+1:,:]
        f[0:pos,:] = input[ind-pos:ind,:]
    else:
        f[0:pos,:] = input[ind-pos:ind,:]
        f[pos+1:,:] = input[ind+1:ind+pos+1,:]
    return f

def seq_frame_word(input,n):
    frame = seq_frame(input,0,n).reshape(1,600)
    for i in range(1,len(input)):
        frame_aux = seq_frame(input,i,n).reshape(1,600)
        frame = np.concatenate([frame,frame_aux])
    return frame

def to_string(array):
    str_list = []
    for i in range(0,len(array)):
        str_list.append(str(int(array[i])))
    return str_list

def verificar(observacao):
    resultados = np.zeros(5)
    lista = ["avance","direita",'esquerda','pare','recue']
    resultados[0] = sum(modelo_avance.score(observacao))
    resultados[1] = sum(modelo_direita.score(observacao))
    resultados[2] = sum(modelo_esquerda.score(observacao))
    resultados[3] = sum(modelo_pare.score(observacao))
    resultados[4] = sum(modelo_recue.score(observacao))
    if np.max(resultados) == 0:
        return 'Nao identificou'

    return lista[np.argmax(resultados)]

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



def evaluate_lenet5(learning_rate=0.1, n_epochs=60,
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

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print(test_set_x.shape.eval())
    tes1 = theano.shared(np.asarray(seq_frame_word(extrair('base/avance_l.wav'),15),dtype=theano.config.floatX),borrow=True)
    tes2 = theano.shared(np.asarray(seq_frame_word(extrair('base/direita_l.wav'),15),dtype=theano.config.floatX),borrow=True)
    tes3 = theano.shared(np.asarray(seq_frame_word(extrair('base/esquerda_l.wav'),15),dtype=theano.config.floatX),borrow=True)
    tes4 = theano.shared(np.asarray(seq_frame_word(extrair('base/pare_l.wav'),15),dtype=theano.config.floatX),borrow=True)

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

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 15, 40))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (15-3+1, 40-3+1) = (13, 38)
    # maxpooling reduces this further to (13/2, 38/2) = (6,19)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 6, 19)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 15, 40),
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (6-3+1, 19-3+1) = (4, 16)
    # maxpooling reduces this further to (4/2, 17/2) = (2,8)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 2, 7)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 6, 19),
        filter_shape=(nkerns[1], nkerns[0], 3, 3),
        poolsize=(2, 2)
    )


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 20300) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    l2_out = 250

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 2 * 8,
        n_out=l2_out,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=l2_out, n_out=16)

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
    while (epoch < n_epochs):

        print(epoch)
        epoch = epoch + 1
        lista = np.arange(0,18350)#29000)
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
        print('Teste Elocução')
        ultra_test =  [
                        model_predict1(i)
                        for i in range(int(ntes1))
                   ]
        askda = np.array(ultra_test).ravel()
        for j in range(len(askda)):
            print(fonemas[askda[j]],',',end='')
        print('a1')
        ultra_test =  [
                        model_predict2(i)
                        for i in range(int(ntes2))
                   ]
        askda = np.array(ultra_test).ravel()
        for j in range(len(askda)):
            print(fonemas[askda[j]],',',end='')
        print('a1')

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
    with open('rede.pkl', 'wb') as f:
        pickle.dump(layer3, f)

    print('Modelo foi salvo com sucesso')
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
