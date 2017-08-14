'''
Created on 25/08/2014

@author: Rafael
'''
from sklearn import svm
import csv
import datetime
import numpy as np
import mfcc
import dill
import HMMCNN as hmm
import scipy.io.wavfile as wave
import random

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

def verificar_gmm(observacao):
    resultados = np.zeros(5)
    lista = ["avance","direita",'esquerda','pare','recue']
    resultados[0] = sum(modelo_avance.score(observacao))
    resultados[1] = sum(modelo_direita.score(observacao))
    resultados[2] = sum(modelo_esquerda.score(observacao))
    resultados[3] = sum(modelo_pare.score(observacao))
    resultados[4] = sum(modelo_recue.score(observacao))
    if np.max(resultados) == 0:
        return 'Nao identificou'
    ind = np.argmax(resultados)
    return lista[ind],ind, resultados

def index_quadro(tempo,tamanho,overlap):
    quadros = [tamanho]
    while quadros[-1] < tempo:
        quadros.append(quadros[-1] + (tamanho-overlap))

    if (quadros[-1] - tempo) <= ((tamanho - overlap) / 2):
        return len(quadros) - 1
    else:
        return len(quadros) - 2

def montar_label(palavra,tamanho,overlap):
    lista = np.array([])
    for i in range(len(palavra)):
        inicio = index_quadro(float(palavra[i]['inicio']),tamanho,overlap) + 1
        fim = index_quadro(float(palavra[i]['fim']),tamanho,overlap)
        lista = np.concatenate([lista,np.repeat(palavra[i]['fonema'],fim-inicio+1)])
    return lista


n = 15
def id_fonema(f):
    fonemas = np.array(["a","v","an","c","i","p","r","rr","e","k","u","d","e_c","t","s","sil"])
    return int(np.where(fonemas == f)[0][0])

with open('modelo_avance_cnn.pkl', 'rb') as f:
    modelo_avance = dill.load(f)
with open('modelo_recue_cnn.pkl', 'rb') as f:
    modelo_recue = dill.load(f)
with open('modelo_pare_cnn.pkl', 'rb') as f:
    modelo_pare = dill.load(f)
with open('modelo_direita_cnn.pkl', 'rb') as f:
    modelo_direita = dill.load(f)
with open('modelo_esquerda_cnn.pkl', 'rb') as f:
    modelo_esquerda = dill.load(f)


lista = []

for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_1_50_avance_',str(i),'.wav']))
for i in range(1,9):
    lista.append( ''.join(['babble2/MeuProjeto_H_1_50_direita_',str(i),'.wav']))
for i in range(1,9):
    lista.append( ''.join(['babble2/MeuProjeto_H_1_50_esquerda_',str(i),'.wav']))
for i in range(1,9):
    lista.append( ''.join(['babble2/MeuProjeto_H_1_50_pare_',str(i),'.wav']))
for i in range(1,9):
    lista.append( ''.join(['babble2/MeuProjeto_H_1_50_recue_',str(i),'.wav']))

for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_2_24_avance_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_2_24_direita_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_2_24_esquerda_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_2_24_pare_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_2_24_recue_',str(i),'.wav']))

for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_3_27_avance_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_3_27_direita_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_3_27_esquerda_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_3_27_pare_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_3_27_recue_',str(i),'.wav']))

for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_4_25_avance_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_4_25_direita_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_4_25_esquerda_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_4_25_pare_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_4_25_recue_',str(i),'.wav']))

for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_5_43_avance_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_5_43_direita_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_5_43_esquerda_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_5_43_pare_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_5_43_recue_',str(i),'.wav']))

for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_6_38_avance_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_6_38_direita_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_6_38_esquerda_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_6_38_pare_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_H_6_38_recue_',str(i),'.wav']))

for i in range(1,9):
    lista.append( ''.join(['babble2/MeuProjeto_M_1_21_avance_',str(i),'.wav']))
for i in range(1,9):
    lista.append( ''.join(['babble2/MeuProjeto_M_1_21_direita_',str(i),'.wav']))
for i in range(1,9):
    lista.append( ''.join(['babble2/MeuProjeto_M_1_21_esquerda_',str(i),'.wav']))
for i in range(1,10):
    lista.append( ''.join(['babble2/MeuProjeto_M_1_21_recue_',str(i),'.wav']))

for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_M_2_25_avance_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_M_2_25_direita_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_M_2_25_esquerda_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_M_2_25_pare_',str(i),'.wav']))
for i in range(1,11):
    lista.append( ''.join(['babble2/MeuProjeto_M_2_25_recue_',str(i),'.wav']))

def verificar(observacao):
    resultados = np.zeros(5)
    lista = ["avance","direita",'esquerda','pare','recue']
    resultados[0] = modelo_avance.forward(observacao)[1]
    resultados[1] = modelo_direita.forward(observacao)[1]
    resultados[2] = modelo_esquerda.forward(observacao)[1]
    resultados[3] = modelo_pare.forward(observacao)[1]
    resultados[4] = modelo_recue.forward(observacao)[1]


    ind = np.argmax(resultados)
    return lista[ind],ind, resultados

lista_comandos = ["avance","direita",'esquerda','pare','recue']

     
erros = {'avance':0,'direita':0,'esquerda':0,'pare':0,'recue':0}
acertos = {'avance':0,'direita':0,'esquerda':0,'pare':0,'recue':0}

quantidade_total = len(lista)
inicio = datetime.datetime.now()
data_csv = []
for c in range(5):
    for i in range(len(lista)):
        #print(lista[i])
        sinal = seq_frame_word(extrair(lista[i]),n)
        #print(lista[i])
        resultado, ind, prob = verificar(sinal)
        print(c, resultado,' - ',classe(lista[i]))
        soma = np.sum(prob)
        positivo = prob[c] / soma
        negativo = 1 - positivo
        #print(resultado, ind, prob, positivo, soma, negativo)
        if (lista[i].find(lista_comandos[c]) > 0):
            data_csv.append([1,positivo])
        else:
            data_csv.append([-1,positivo])

        if (lista[i].find(resultado) > 0):
            acertos[resultado] += 1
            print(i,' de ', quantidade_total, ' - acertou. ')
        else:
            erros[classe(lista[i])] += 1
            print(i,' de ', quantidade_total, ' - errou. ')

fim = datetime.datetime.now()
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
print('tempo decorrido: ', fim - inicio)

with open('babble_independente.csv', 'w', newline='') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data_csv)
