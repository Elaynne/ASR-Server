'''
Created on 27/09/2016

@author: Luciana
'''

import numpy as np
import dill
import Carrega_dados as cd
from datetime import datetime
import utils as utils
import theano as th
from nnet.neuralnet import NeuralNet
import aux
import utils_treinamento as ut

start_time = datetime.now()

rodada = 10

with open('base_BF_rod' + str(rodada) + '.pkl', 'rb') as f:
    base = dill.load(f)

lista, y_train = base[0],base[2]

dados_carregado = cd.Carrega()

args = utils.read_args('configs/default.ast')
print(args)
num_epochs, nnet_args = args['num_epochs'], args['nnet_args']
chars = utils.mapeamento_palavra()
num_classes = len(chars)

num_samples = len(lista)
printer = utils.Printer(chars)


data_x, data_y = ut.formata_padrao_entrada_saida(num_classes, y_train, dados_carregado, lista, largura)
img_ht = data_x[0].shape[0]

print('\nInput Dim: {}'
      '\nNum Classes: {}'
      '\nNum Samples: {}'
      '\nNum Epochs: {}'
      '\nFloatX: {}'
      '\n'.format(img_ht, num_classes, num_samples, num_epochs, th.config.floatX))


data_x,data_y = utils.amostras_shuffle(data_x,data_y)

print('Construindo a rede')
ntwk = NeuralNet(img_ht, num_classes, **nnet_args)
print(ntwk)

try:
    for epoch in range(num_epochs):

        tott = (datetime.now()) - start_time
        secs = tott.seconds
        hours, minutes = tott.days * 24 + tott.seconds // 3600, (tott.seconds // 60) % 60
        for samp in range((num_samples)):
            x = data_x[samp]
            y = data_y[samp]
            cst, pred, aux = ntwk.trainer(x, y)
        if (epoch == 25 ):
            ntwk.salvar('BLSTM_' + str(rodada) + '_' + str(epoch))

except (KeyboardInterrupt, SystemExit):
    print('salvou')


end_time = datetime.now()

tott = end_time - start_time
secs = tott.seconds
hours, minutes = tott.days * 24 + tott.seconds // 3600, (tott.seconds // 60) % 60

ntwk.salvar('BLSTM')

secs %= 60
print('\n Tempo : {}:{:02d}:{:02d}'.format(
    hours, minutes, secs))


