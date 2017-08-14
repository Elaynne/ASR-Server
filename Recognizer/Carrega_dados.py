'''
Created on 14/07/2016

@author: Luciana
'''
import numpy
import random
import dill
import theano
import theano.tensor as T
import pickle
import utils

class Carrega:

    def __init__(self):
        with open('rede.pkl', 'rb') as f:
            self.rede = pickle.load(f)
            _, self.largura, self.rotulo = utils.leitura_arquivo(0)
            self.linha_total = self.largura * 40
            # print('CNN , rodada:', str(rodada), ' , largura: ', self.largura), ' , rotulo: ', self.rotulo , ' , tamLinhaTotal: ' , self.linha_total

    def calcular_probabilidades(self, entrada):
        batch_size = 100

        linhas = entrada.shape[0]
        index = T.lscalar()

        x = self.rede.salvation

        base = numpy.concatenate([entrada,numpy.zeros([batch_size - (linhas % batch_size),self.linha_total])])
        rep_for = base.shape[0] / batch_size


        base = theano.shared(numpy.asarray(base,dtype=theano.config.floatX),borrow=True)
        theano.config.exception_verbosity = 'high'

        predicao = theano.function([index], self.rede.p_y_given_x,
         givens={\
            x: base[index * batch_size: (index + 1) * batch_size]}, on_unused_input='ignore')

        probabilidades = predicao(0).reshape(batch_size,self.rotulo)

        for i in range(1,int(rep_for)):
            aux =  predicao(i)
            probabilidades = numpy.concatenate([probabilidades, aux.reshape(batch_size,self.rotulo)])

        return probabilidades[0:linhas,:]





    def treinar(self, obs):
        observacoes = self.calcular_probabilidades(obs)
        return observacoes













