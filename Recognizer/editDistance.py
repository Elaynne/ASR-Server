# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 19:49:41 2016

@author: luciana
"""

import numpy as np

def edit_distance(ref,hyp):
    n = len(ref)
    m = len(hyp)

    ins = dels = subs = corr = 0
    
    D = np.zeros((n+1,m+1))

    D[:,0] = np.arange(n+1)
    D[0,:] = np.arange(m+1)

    for i in  range(1,n+1):
        for j in  range(1,m+1):
            if ref[i-1] == hyp[j-1]:
                D[i,j] = D[i-1,j-1]
            else:
                D[i,j] = min(D[i-1,j],D[i,j-1],D[i-1,j-1])+1

    i=n
    j=m
    while i>0 and j>0:
        if ref[i-1] == hyp[j-1]:
            corr += 1
        elif D[i-1,j] == D[i,j]-1:
            ins += 1
            j += 1
        elif D[i,j-1] == D[i,j]-1:
            dels += 1
            i += 1
        elif D[i-1,j-1] == D[i,j]-1:
            subs += 1
        i -= 1
        j -= 1

    ins += i
    dels += j

    return D[-1,-1],ins,dels,subs,corr
#PARE - ins = 0, del = 0, subs = 2, corr = 2
#Recue - ins = 1, del =0, subs = 1, corr = 3
def disp(ref,hyp):
    dist,ins,dels,subs,corr = edit_distance(ref,hyp)
    # print(str(dist) + " insercao =  " + str(ins) + " deleções =  " + str(dels) + " subs =  " + str(subs))
    return dist


def disp2(ref,hyp):
    dist,ins,dels,subs,corr = edit_distance(ref,hyp)
    return dist,subs
#    lista = ["avance", "direita", 'esquerda', 'pare', 'recue']

def empate(resultados,lista, saida_obtida):
    prim = np.where(resultados == resultados.min())[0][0]
    seg = np.where(resultados == resultados.min())[0][1]
    dis1,subs1  = disp2(lista[prim],saida_obtida)
    dis2, subs2 = disp2(lista[seg], saida_obtida)
    if(subs1 < subs2):
        return prim
    else:
        return seg


if __name__=="__main__":

    ref = list('avance')
    hyp = list('navanci')
    disp(ref,hyp)

    ref = list('direita')
    disp(ref, hyp)

    ref = list('esquerda')
    disp(ref, hyp)


    ref = list('pare')
    disp(ref,hyp)

    ref = list('recue')
    disp(ref, hyp)