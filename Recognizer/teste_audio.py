'''
Created on 14/07/2016

@author: Luciana
'''
import numpy as np
import dill
import Carrega_dados as cd
import utils as utils
import theano as th
import aux as aux
import editDistance as ed
from nnet.neuralnet import neuralNetTest
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from scipy import signal


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


largura = 70
epoca = 60
lista = []

def retorna_logfbank(arquivo, nfilt = 40):
    (rate,sig) = wav.read(arquivo)

    if(rate  != 8000):
        rate = 8000
        sig = signal.resample(sig,rate)

    return logfbank(sig,rate, nfilt=nfilt)


def teste(arquivo):
    with open('BLSTM.pkl', 'rb') as pkl_file:
        layer2,layer1,image = dill.load(pkl_file)


    audio = retorna_logfbank(arquivo)
    dados_carregado = cd.Carrega()

    chars = utils.mapeamento_palavra()

    printer = utils.Printer(chars)

    data_x = []
    _, sinal = aux.deslocamento_amostra(audio, larg=largura)
    sinal = np.concatenate(sinal)
    probs = dados_carregado.treinar(sinal)

    data_x.append(np.asarray(probs.T, dtype=th.config.floatX))

    ntwk = neuralNetTest(layer2,layer1,image)


    x = data_x[0]
    saida_obtida, _ = ntwk.tester(x)
    rotulo_pred, retornado2 = printer.rotulo_(saida_obtida)

    resultado = verificar(retornado2)

    print("Palavra encontrada:", resultado)
    return resultado




caminho_arquivo_audio = "avance5.wav"
teste(caminho_arquivo_audio)
