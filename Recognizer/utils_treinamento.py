# -*- coding: utf-8 -*-
import numpy as np
import utils
import theano as th
import aux
import dill
import editDistance as ed




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



def formata_padrao_entrada_saida(num_classes, y_train, dados_carregado, lista, largura=70):
    data_x, data_y = [], []

    for indice in range(len(lista)):
        y = utils.classe(y_train[indice])  # Recupera a palavra
        y = utils.palavra_indice(y)
        y1 = utils.insere_blanks(y, num_classes)
        data_y.append(np.asarray(y1, dtype=np.int32))
        _, sinal = aux.deslocamento_amostra(lista[indice], larg=largura)
        sinal = np.concatenate(sinal)
        probs = dados_carregado.treinar(sinal)

        data_x.append(np.asarray(probs.T, dtype=th.config.floatX))

    return data_x, data_y


def testar(ntwk, rodada, dados_carregado, largura):
    with open('base_BF_rod' + str(rodada) + '.pkl', 'rb') as f:
        base = dill.load(f)

    lista, y_train = base[0], base[2]


    args = utils.read_args('configs/default.ast')
    num_epochs, nnet_args = args['num_epochs'], args['nnet_args']
    chars = utils.mapeamento_palavra()
    num_classes = len(chars)

    num_samples = len(lista)
    printer = utils.Printer(chars)

    data_x, data_y = formata_padrao_entrada_saida(num_classes, y_train, dados_carregado, lista, largura)

    img_ht = data_x[0].shape[0]


    acertos = 0
    erros = 0
    quantidade_total = len(lista)

    for c in range(1):
        for i in range(len(lista)):
            # print(lista[i])
            x = data_x[i]
            y = data_y[i]

            _, esperado = printer.yprint2(y)
            saida_obtida, _ = ntwk.tester(x)
            rotulo_pred, retornado2 = printer.rotulo_(saida_obtida)

            resultado = verificar(retornado2)

            if (esperado == resultado):
                acertos+= 1
            else:
                erros+= 1

    return acertos, erros