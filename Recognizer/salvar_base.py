'''
Created on 25/08/2014

@author: Rafael
'''
from sklearn import mixture
import numpy as np
import mfcc
import dill
import scipy.io.wavfile as wave
from pylab import *
import random

def extrair(arquivo):
    rate = 8000
    sinal = wave.read('base/'+arquivo, mmap = 'false')
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



def id_fonema(f):
    fonemas = np.array(["a","v","an","c","i","p","r","rr","e","k","u","d","e_c","t","s","sil"])
    return int(np.where(fonemas == f)[0][0])


tamanho = 0.0250
overlap = 0.010
n = 15


arquivo = list(open('biochaves-lh1.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = word
labels = label

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

arquivo = list(open('biochaves-lh2.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = np.concatenate([frames,word])
labels = np.concatenate([labels,label])

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

arquivo = list(open('biochaves-lh3.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = np.concatenate([frames,word])
labels = np.concatenate([labels,label])

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

arquivo = list(open('biochaves-lh4.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = np.concatenate([frames,word])
labels = np.concatenate([labels,label])

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

arquivo = list(open('biochaves-lh5.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = np.concatenate([frames,word])
labels = np.concatenate([labels,label])

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

arquivo = list(open('biochaves-lh5.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = np.concatenate([frames,word])
labels = np.concatenate([labels,label])

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

arquivo = list(open('biochaves-lm2.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = np.concatenate([frames,word])
labels = np.concatenate([labels,label])

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

with open('base_fbank.pkl', 'wb') as f:
    dill.dump((frames,np.append([],labels)), f)
print('dados mfcc', frames.shape)
print(labels.shape)
print(np.unique(labels))

arquivo = list(open('biochaves-lh6.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = word
labels = label

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

arquivo = list(open('biochaves-lm1.txt','r'))
nomes = []
fonemas = []
for i in range(int(len(arquivo)/2)):
    nomes.append(arquivo[i*2].strip())
    temp = arquivo[i*2+1].strip().split(';')
    f = []
    for j in range(len(temp)):
        fonema = temp[j].split(':')
        tempo = fonema[1].split('-')
        f.append({'fonema':id_fonema(fonema[0]),'inicio':float(tempo[0]),'fim':float(tempo[1])})
    fonemas.append(f)



label = montar_label(fonemas[0],tamanho,overlap)
word = seq_frame_word(extrair(nomes[0]),n)
label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
frames = np.concatenate([frames,word])
labels = np.concatenate([labels,label])

for i in range(1,len(nomes)):
    label = montar_label(fonemas[i],tamanho,overlap)
    word = seq_frame_word(extrair(nomes[i]),n)
    label = np.concatenate([label, np.repeat(id_fonema('sil'),math.fabs(label.shape[0]-word.shape[0]))])
    print(nomes[i],label.shape[0]-word.shape[0],label.shape[0],word.shape[0], (label.shape[0]-word.shape[0]) > 0)
    frames = np.concatenate([frames,word])
    labels = np.concatenate([labels,label])

with open('base_fbank_teste.pkl', 'wb') as f:
    dill.dump((frames,np.append([],labels)), f)
print('dados mfcc teste', frames.shape)
print(labels.shape)
print(np.unique(labels))