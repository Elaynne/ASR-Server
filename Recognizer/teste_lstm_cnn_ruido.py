'''
Created on 14/07/2016

@author: Luciana
'''
import numpy as np
import Carrega_dados as cd
import utils as utils
import theano as th
import aux as aux
import editDistance as ed

from nnet.neuralnet import neuralNetTest

def verificar(saida_obtida):
    resultados = np.zeros(5)
    lista = ["avance", "direita", 'esquerda', 'pare', 'recue']
    resultados[0] = ed.disp(lista[0], saida_obtida)
    resultados[1] = ed.disp(lista[1], saida_obtida)
    resultados[2] = ed.disp(lista[2], saida_obtida)
    resultados[3] = ed.disp(lista[3], saida_obtida)
    resultados[4] = ed.disp(lista[4], saida_obtida)
    #print(resultados)
    if(len(np.where(resultados == resultados.min())[0])>1):
        ind = ed.empate(resultados,lista, saida_obtida)
    else:
        ind = np.argmin(resultados)
    return lista[ind]



lista = []

# nome_arquivo = 'babble2'
# nome_arquivo = 'factory'
# nome_arquivo = 'volvo'
nome_arquivo = 'base'

import dill
with open('BLSTM.pkl', 'rb') as pkl_file:
    layer2,layer1,image = dill.load(pkl_file)

rodada = 10
# largura = 60

with open('base_BF_rod' + str(rodada) + '_' + nome_arquivo + '_teste.pkl', 'rb') as f:
    base = dill.load(f)

lista, y_train = base[0], base[2]

dados_carregado = cd.Carrega()

args = utils.read_args('configs/default.ast')
num_epochs, nnet_args = args['num_epochs'], args['nnet_args']
chars = utils.mapeamento_palavra()
num_classes = len(chars)

num_samples = len(lista)
printer = utils.Printer(chars)

data_x, data_y = [], []

for indice in range(len(lista)):
    y = utils.classe(y_train[indice])  # Recupera a palavra
    y = utils.palavra_indice(y)
    y1 = utils.insere_blanks(y, num_classes)
    data_y.append(np.asarray(y1, dtype=np.int32))
    _, sinal = aux.deslocamento_amostra(lista[indice], larg=70)
    sinal = np.concatenate(sinal)
    probs = dados_carregado.treinar(sinal)

    data_x.append(np.asarray(probs.T, dtype=th.config.floatX))


img_ht = data_x[0].shape[0]

ntwk = neuralNetTest(layer2,layer1,image)





erros = {'avance':0,'direita':0,'esquerda':0,'pare':0,'recue':0}
acertos = {'avance':0,'direita':0,'esquerda':0,'pare':0,'recue':0}

quantidade_total = len(lista)
data_csv = []
for c in range(1):
    for i in range(len(lista)):
        #print(lista[i])
        x = data_x[i]
        y = data_y[i]

        _, esperado = printer.yprint2(y)
        saida_obtida, _ = ntwk.tester(x)
        rotulo_pred, retornado2 = printer.rotulo_(saida_obtida)

        resultado = verificar(retornado2)

        print("obtido :", retornado2)
        print("esperado: ", esperado, ' -obtido: ', resultado)
        if (esperado == resultado):
            acertos[resultado] += 1
            print(i,' de ', quantidade_total, ' - acertou. ')
        else:
            erros[esperado] += 1
            print(i,' de ', quantidade_total, ' - errou. ')


print("----------------Acertos-------------")
print(acertos)
print("----------------Erros-------------")
print(erros)

total_erros = sum(list(erros.values()))
total_acertos = sum(list(acertos.values()))
print('Total acertos')
print(total_acertos,' -', total_acertos/(total_acertos+total_erros)*100,'%')
print('Total erros')
print(total_erros,' -', total_erros/(total_acertos+total_erros)*100,'%')




