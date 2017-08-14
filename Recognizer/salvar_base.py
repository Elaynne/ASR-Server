# -*- coding: utf-8 -*-
import ast
import numpy as np
import dill
from python_speech_features import logfbank
import scipy.io.wavfile as wav



def classe2(arquivo):
    if (arquivo.find('direita') > 0):
        return 1
    if (arquivo.find('esquerda') > 0):
        return 2
    if (arquivo.find('recue') > 0):
        return 4
    if (arquivo.find('pare') > 0):
        return 3
    if (arquivo.find('avance') > 0):
        return 0


def lista_arquivos(nome_arquivo = 'base'):
    inicio = 1
    lista = []
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_1_50_avance_', str(i), '.wav']))
    for i in range(inicio, 9):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_1_50_direita_', str(i), '.wav']))
    for i in range(inicio, 9):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_1_50_esquerda_', str(i), '.wav']))
    for i in range(inicio, 9):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_1_50_pare_', str(i), '.wav']))
    for i in range(inicio, 9):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_1_50_recue_', str(i), '.wav']))

    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_2_24_avance_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_2_24_direita_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_2_24_esquerda_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_2_24_pare_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_2_24_recue_', str(i), '.wav']))

    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_3_27_avance_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_3_27_direita_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_3_27_esquerda_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_3_27_pare_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_3_27_recue_', str(i), '.wav']))

    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_4_25_avance_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_4_25_direita_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_4_25_esquerda_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_4_25_pare_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_4_25_recue_', str(i), '.wav']))

    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_5_43_avance_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_5_43_direita_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_5_43_esquerda_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_5_43_pare_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_5_43_recue_', str(i), '.wav']))

    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_6_38_avance_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_6_38_direita_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_6_38_esquerda_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_6_38_pare_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_H_6_38_recue_', str(i), '.wav']))

    for i in range(inicio, 9):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_1_21_avance_', str(i), '.wav']))
    for i in range(inicio, 9):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_1_21_direita_', str(i), '.wav']))
    for i in range(inicio, 9):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_1_21_esquerda_', str(i), '.wav']))
    for i in range(inicio, 10):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_1_21_recue_', str(i), '.wav']))

    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_2_25_avance_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_2_25_direita_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_2_25_esquerda_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_2_25_pare_', str(i), '.wav']))
    for i in range(inicio, 11):
        lista.append(''.join([nome_arquivo + '/MeuProjeto_M_2_25_recue_', str(i), '.wav']))
    return lista

def retorna_logfbank(arquivo, nfilt = 40):
    (rate,sig) = wav.read(arquivo)
    print('rate: ' , rate)
    return logfbank(sig,rate, nfilt=nfilt)


def carrega_arquivos(pasta = 'base', rodada = 1):
    entrada = list()
    saida = list()
    arqs = list()
    caminhos = lista_arquivos()
    for arquivo in caminhos:
        if(arquivo.find('_'+str(rodada)+'.wav') == -1):
            entrada.append(retorna_logfbank('../'+arquivo))
            saida.append(classe2(arquivo))
            arqs.append((arquivo))
    return entrada, saida, arqs

def carrega_arquivos_teste(pasta = 'babble2', rodada = 1):
    entrada = list()
    saida = list()
    arqs = list()
    caminhos = lista_arquivos(pasta)
    for arquivo in caminhos:
        if(arquivo.find('_'+str(rodada)+'.wav') > -1):
            entrada.append(retorna_logfbank('../'+arquivo))
            saida.append(classe2(arquivo))
            arqs.append(arquivo)
    return entrada, saida, arqs


def salvar_bancos_filtro(pasta = 'base', rodada = 1):

    if(pasta == 'base'):
        x, y, ar = carrega_arquivos(pasta = 'base', rodada=rodada)
        x = normalizaTransponhe(x)
        with open('base/base_BF_rod'+str(rodada)+'.pkl', 'wb') as f:
            dill.dump((x, np.append([], y), ar), f)

    x, y, ar = carrega_arquivos_teste(pasta=pasta, rodada=rodada)
    x = normalizaTransponhe(x)
    print('PASTA: ' , pasta)
    with open('base/base_BF_rod'+str(rodada)+'_'+pasta+'_teste.pkl', 'wb') as f:
        dill.dump((x, np.append([], y), ar), f)


def normalizaTransponhe(x):
    x = normaliza(x)
    x = [np.transpose(a) for a in x]
    return x


def normaliza(X_train):
    #Pegar Máximo e mínimo
    maximo = np.max( [ np.max(x1) for x1 in X_train] )
    minimo = np.min( [ np.min(x1) for x1 in X_train] )
    for i in range(len(X_train)):
        for ii in range(len(X_train[i])):
            X_train[i][ii] = (X_train[i][ii] - minimo) / (maximo - minimo)
    return X_train

rodada = 7
# salvar_bancos_filtro('base', rodada)
# salvar_bancos_filtro('babble2', rodada)
# salvar_bancos_filtro('factory', rodada)
# salvar_bancos_filtro('volvo', rodada)


