# -*- coding: utf-8 -*-
import ast
import numpy as np
import pickle



def leitura_arquivo(largura):
    arq = open('arquivo/faixas.txt' , 'rb')
    texto = pickle.load(arq)
    # print(texto)
    arq.close()
    return texto['faixas'] , texto['largura'] , texto['rotulos']


def mapeamento_palavra():
    #avance direita esquerda pare recue
    chars = "acdeinpqrstuv"
    mapping = {c: i for i, c in enumerate(chars)}
    lista = list()
    for char in chars:
        lista.append(char)
    return lista


def insere_blanks(y, blank):
    # Insert blanks at alternate locations in the labelling (blank is blank)
    y1 = [blank]
    for char in y:
        y1 += [char, blank]
    return y1

def palavra_indice( palavra):
    letras = np.array(["a", "c", "d", "e", "i", "n", "p", "q", "r", "s", "t", "u", "v"])
    labels_list = [int(np.where(letras == l)[0][0]) for l in palavra]
    return labels_list


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






class Printer():
    def __init__(self, chars):
        """
        Creates a function that can print a predicted output of the CTC RNN
        It removes the blank characters (need to be set to n_classes),
        It also removes duplicates
        :param list chars: list of characters
        """
        self.chars = chars + ['blank']
        self.n_classes = len(self.chars) - 1

    def yprint(self, labels):
        labels_out = []
        for il, l in enumerate(labels):
            if (l != self.n_classes) and (il == 0 or l != labels[il-1]):
                labels_out.append(l)
        labels_list = [self.chars[l] for l in labels_out]
        # print(labels_out, ' '.join(labels_list))
        return labels_out, labels_list


    def yprint2(self, labels):
        labels_out = []
        for il, l in enumerate(labels):
            if (l != self.n_classes) and (il == 0 or l != labels[il-1]):
                labels_out.append(l)
        labels_list = [self.chars[l] for l in labels_out]
        labels_list = ''.join(labels_list)
        return labels_out, labels_list

    def ylen(self, labels):
        length = 0
        for il, l in enumerate(labels):
            if (l != self.n_classes) and (il == 0 or l != labels[il-1]):
                length += 1
        return length



    def palavra_indice(self, palavra):
        letras = np.array(["a","c","d","e","i","n","p","q","r","s","t","u","v"])
        labels_list = [int(np.where(letras == l)[0][0]) for l in palavra]
        return labels_list



    def show_all(self, shown_seq, shown_img,
                 softmax_firings=None):
        """
        Utility function to show the input and output and debug
        :param shown_seq: Labelings of the input
        :param shown_img: Input Image
        :param softmax_firings: Seen Probabilities (Excitations of Softmax)
        :param aux_imgs: List of pairs of images and names
        :return:
        """
        print('Desejado : ', end='')
        labels, labels_chars = self.yprint(shown_seq)

        if softmax_firings is not None:
            print('Obtido   : ', end='')
            maxes = np.argmax(softmax_firings, 0)
            labels_y, labels_char_y = self.yprint(maxes)

        return labels_chars,labels_char_y

    def rotulo_(self,pred):
        maxes  = np.argmax(pred,0)

        label, _  = self.yprint(maxes)

        _,palavra_gerada = self.yprint2(maxes)
        return label, palavra_gerada




def read_args(default='../configs/default.ast'):
    with open(default, 'r') as dfp:
        args = ast.literal_eval(dfp.read())

    return args






