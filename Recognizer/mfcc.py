# -*- coding: utf-8 -*-
import csv
import numpy
import io
from scipy.fftpack import dct
from scipy.fftpack import fft
import math
import cmath
import matplotlib.pylab as plt
import scipy.io.wavfile as wav

class mfcc:

    def __init__(self, quadro_amostra, taxa_amostra, num_cepstrais):
        self.quadro_amostra = quadro_amostra
        self.taxa_amostra = taxa_amostra
        self.num_cepstrais = num_cepstrais
        self.freq_filtro_alto = taxa_amostra / 2.0
        self.freq_filtro_baixo = 80.00
        self.pre_alpha = 0.95
        self.num_filtros_mel = 40
        self.quadro = 0.025
        self.overlap = 0.010
        #magnitude do espectro
        self.bin = []
        self.bin_quadros = []

    def extrair_mfcc(self, sinal):

        #Quantidade e tamanho de quadros
        sinal = numpy.array(sinal,dtype=numpy.int32)
        sinal = numpy.append(sinal[0],sinal[1:]-self.pre_alpha*sinal[:-1])
        tamanho_sinal = len(sinal)
        #tamanho_quadro = self.quadro * self.taxa_amostra
        calc_quadro = (len(sinal) + 5600) / 568000
        tamanho_quadro = calc_quadro * self.taxa_amostra


        tamanho_overlap = self.overlap * self.taxa_amostra
        tamanho_quadro = int(round(tamanho_quadro))
        tamanho_overlap = int(round(tamanho_overlap))
#         if tamanho_sinal <= tamanho_quadro:
#             num_quadros = 1
#         else:
#             num_quadros = 1 + int(math.ceil((1.0*tamanho_sinal - tamanho_quadro)/tamanho_overlap))

        contador = tamanho_quadro
        num_quadros = 1
        while(contador < tamanho_sinal):
            num_quadros += 1
            contador -= tamanho_overlap
            contador += tamanho_quadro

        #Completar zeros
        preencher = int((num_quadros)*(tamanho_quadro- tamanho_overlap)+ tamanho_overlap)
        zeros = numpy.zeros((preencher - tamanho_sinal,))
        sinal_preenchido = numpy.concatenate((sinal,zeros))
        indices = numpy.tile(numpy.arange(0,tamanho_quadro),(num_quadros,1)) + numpy.tile(numpy.arange(0,(num_quadros)*(tamanho_quadro- tamanho_overlap),(tamanho_quadro- tamanho_overlap)),(tamanho_quadro,1)).T
        indices = numpy.array(indices,dtype=numpy.int32)
        quadros = sinal_preenchido[indices]
        janela = numpy.tile(numpy.ones(tamanho_quadro),(num_quadros,1))
        quadros = quadros * janela

        #hamming window
        hamming = numpy.zeros(tamanho_quadro)
        for i in range(tamanho_quadro):
            hamming[i] = 0.54 - (0.46* numpy.cos(2*i*numpy.pi/(tamanho_quadro-1)))
        for i in range(0,len(quadros)):
            quadros[i] = numpy.multiply(quadros[i],hamming)

        self.bin_quadros = numpy.zeros((len(quadros),self.num_filtros_mel))

        for q in range(0, len(quadros)):
            #aplicar dft

            self.bin = fft(quadros[q])
            self.quadro_amostra = len(quadros[q])

    #         N = len(sinal)
    #         self.bin = numpy.zeros(N)
    #         for k in range(0,N):
    #             for n in range(0,N):
    #                 self.bin[k] = self.bin[k] + sinal[n] * numpy.exp(-2j * (math.pi / N)* k * n)
    #             self.bin[k] = self.bin[k] / numpy.sqrt(N)

            #Calculo do valor absoluto
            for i in range(0,len(self.bin)):
                self.bin[i] = math.sqrt(self.bin[i].real * self.bin[i].real + self.bin[i].imag * self.bin[i].imag)

            #Enfatizar sinais de alta frequencia
            for i in range(1,len(self.bin)):
                self.bin[i] = self.bin[i] - (self.pre_alpha * self.bin[i])

            #Processo de Filtragem
            #Preparar filtro
            cbin = numpy.zeros(self.num_filtros_mel + 2)
            cbin[0] = (round(self.freq_filtro_baixo / self.taxa_amostra * self.quadro_amostra))
            cbin[len(cbin)- 1] = (self.quadro_amostra / 2)
            for i in range(1,self.num_filtros_mel+1):
                mel_baixa = 2595 * math.log10(1 + self.freq_filtro_baixo / 700);
                mel_alta = 2595 * math.log10(1 + self.freq_filtro_alto / 700);
                temp = mel_baixa + ((mel_alta - mel_baixa) / (self.num_filtros_mel + 1)) * i
                centro = 700 * (math.pow(10, temp / 2595)- 1)
                cbin[i] = round(centro / self.taxa_amostra * self.quadro_amostra)

            #Processar banco de filtros Mel
            temp = numpy.zeros(self.num_filtros_mel + 2)
            for k in range(1,self.num_filtros_mel+1):
                num1 = 0
                num2 = 0
                i = cbin[k-1]
                while i <= cbin[k]:
                    num1 = num1 + ((i - cbin[(k - 1)] + 1) / (cbin[k] - cbin[(k - 1)] + 1)) * self.bin[i]
                    i += 1
                i = cbin[k] + 1
                while i <= cbin[k + 1]:
                    num2 += (1 - ((i - cbin[k]) / (cbin[k + 1] - cbin[k] + 1))) * self.bin[i]
                    i += 1
                temp[k] = num1 + num2
            fbank = numpy.zeros(self.num_filtros_mel)
            for i in range(0,self.num_filtros_mel):
                fbank[i] = temp[i + 1]

            #Transformação não linear
            f = numpy.zeros(len(fbank))
            floor = -50

            for i in range(0,len(fbank)):
                f[i] = numpy.log(fbank[i]);
                if f[i] < floor:
                    f[i] = floor;

            #Coeficientes cepstrais através da transformada do cosseno
            self.bin_quadros[q] = dct(f, type=2, axis=0, norm='ortho')#[:,:self.num_cepstrais]

#             N = len(self.bin_quadros[0])
#             self.bin = numpy.zeros(N)
#             for k in range(0,N):
#                 alpha = 0
#                 if (k == 0):
#                     alpha = numpy.sqrt(1/N)
#                 else:
#                     alpha = numpy.sqrt(2/N)
#
#                 for n in range(0,N):
#                     self.bin[k] = self.bin[k] + f[n] * numpy.cos((numpy.pi*k*(2*n + 1))/ 2 * N)
#
#                 self.bin[k] = alpha * self.bin[k]
#             self.bin_quadros[q] = self.bin
        return self.bin_quadros
    def extrair_fbank_mel(self, sinal):

        #Quantidade e tamanho de quadros
        sinal = numpy.array(sinal,dtype=numpy.int32)
        sinal = numpy.append(sinal[0],sinal[1:]-self.pre_alpha*sinal[:-1])
        tamanho_sinal = len(sinal)
        tamanho_quadro = self.quadro * self.taxa_amostra
        tamanho_overlap = self.overlap * self.taxa_amostra
        tamanho_quadro = int(round(tamanho_quadro))
        tamanho_overlap = int(round(tamanho_overlap))
#         if tamanho_sinal <= tamanho_quadro:
#             num_quadros = 1
#         else:
#             num_quadros = 1 + int(math.ceil((1.0*tamanho_sinal - tamanho_quadro)/tamanho_overlap))

        contador = tamanho_quadro
        num_quadros = 1
        while(contador < tamanho_sinal):
            num_quadros += 1
            contador -= tamanho_overlap
            contador += tamanho_quadro

        #Completar zeros
        preencher = int((num_quadros)*(tamanho_quadro- tamanho_overlap)+ tamanho_overlap)
        zeros = numpy.zeros((preencher - tamanho_sinal,))
        sinal_preenchido = numpy.concatenate((sinal,zeros))
        indices = numpy.tile(numpy.arange(0,tamanho_quadro),(num_quadros,1)) + numpy.tile(numpy.arange(0,(num_quadros)*(tamanho_quadro- tamanho_overlap),(tamanho_quadro- tamanho_overlap)),(tamanho_quadro,1)).T
        indices = numpy.array(indices,dtype=numpy.int32)
        quadros = sinal_preenchido[indices]
        janela = numpy.tile(numpy.ones(tamanho_quadro),(num_quadros,1))
        quadros = quadros * janela

        #hamming window
        hamming = numpy.zeros(tamanho_quadro)
        for i in range(tamanho_quadro):
            hamming[i] = 0.54 - (0.46* numpy.cos(2*i*numpy.pi/(tamanho_quadro-1)))
        for i in range(0,len(quadros)):
            quadros[i] = numpy.multiply(quadros[i],hamming)

        self.bin_quadros = numpy.zeros((len(quadros),self.num_filtros_mel))

        for q in range(0, len(quadros)):
            #aplicar dft

            self.bin = fft(quadros[q])
            self.quadro_amostra = len(quadros[q])

    #         N = len(sinal)
    #         self.bin = numpy.zeros(N)
    #         for k in range(0,N):
    #             for n in range(0,N):
    #                 self.bin[k] = self.bin[k] + sinal[n] * numpy.exp(-2j * (math.pi / N)* k * n)
    #             self.bin[k] = self.bin[k] / numpy.sqrt(N)

            #Calculo do valor absoluto
            for i in range(0,len(self.bin)):
                self.bin[i] = math.sqrt(self.bin[i].real * self.bin[i].real + self.bin[i].imag * self.bin[i].imag)

            #Enfatizar sinais de alta frequencia
            for i in range(1,len(self.bin)):
                self.bin[i] = self.bin[i] - (self.pre_alpha * self.bin[i])

            #Processo de Filtragem
            #Preparar filtro
            cbin = numpy.zeros(self.num_filtros_mel + 2)
            cbin[0] = (round(self.freq_filtro_baixo / self.taxa_amostra * self.quadro_amostra))
            cbin[len(cbin)- 1] = (self.quadro_amostra / 2)
            for i in range(1,self.num_filtros_mel+1):
                mel_baixa = 2595 * math.log10(1 + self.freq_filtro_baixo / 700);
                mel_alta = 2595 * math.log10(1 + self.freq_filtro_alto / 700);
                temp = mel_baixa + ((mel_alta - mel_baixa) / (self.num_filtros_mel + 1)) * i
                centro = 700 * (math.pow(10, temp / 2595)- 1)
                cbin[i] = round(centro / self.taxa_amostra * self.quadro_amostra)

            #Processar banco de filtros Mel
            temp = numpy.zeros(self.num_filtros_mel + 2)
            for k in range(1,self.num_filtros_mel+1):
                num1 = 0
                num2 = 0
                i = cbin[k-1]
                while i <= cbin[k]:
                    num1 = num1 + ((i - cbin[(k - 1)] + 1) / (cbin[k] - cbin[(k - 1)] + 1)) * self.bin[i]
                    i += 1
                i = cbin[k] + 1
                while i <= cbin[k + 1]:
                    num2 += (1 - ((i - cbin[k]) / (cbin[k + 1] - cbin[k] + 1))) * self.bin[i]
                    i += 1
                temp[k] = num1 + num2
            fbank = numpy.zeros(self.num_filtros_mel)
            for i in range(0,self.num_filtros_mel):
                fbank[i] = temp[i + 1]

            #Transformação não linear
            f = numpy.zeros(len(fbank))
            floor = -50

            for i in range(0,len(fbank)):
                f[i] = numpy.log(fbank[i]);
                if f[i] < floor:
                    f[i] = floor;
            self.bin_quadros[q] = f

        return self.bin_quadros

    def extrair_espectograma(self, sinal):

        #Quantidade e tamanho de quadros
        sinal = numpy.array(sinal,dtype=numpy.int32)
        sinal = numpy.append(sinal[0],sinal[1:]-self.pre_alpha*sinal[:-1])
        tamanho_sinal = len(sinal)
        tamanho_quadro = self.quadro * self.taxa_amostra
        tamanho_overlap = self.overlap * self.taxa_amostra
        tamanho_quadro = int(round(tamanho_quadro))
        tamanho_overlap = int(round(tamanho_overlap))
#         if tamanho_sinal <= tamanho_quadro:
#             num_quadros = 1
#         else:
#             num_quadros = 1 + int(math.ceil((1.0*tamanho_sinal - tamanho_quadro)/tamanho_overlap))

        contador = tamanho_quadro
        num_quadros = 1
        while(contador < tamanho_sinal):
            num_quadros += 1
            contador -= tamanho_overlap
            contador += tamanho_quadro

        #Completar zeros
        preencher = int((num_quadros)*(tamanho_quadro- tamanho_overlap)+ tamanho_overlap)
        zeros = numpy.zeros((preencher - tamanho_sinal,))
        sinal_preenchido = numpy.concatenate((sinal,zeros))
        indices = numpy.tile(numpy.arange(0,tamanho_quadro),(num_quadros,1)) + numpy.tile(numpy.arange(0,(num_quadros)*(tamanho_quadro- tamanho_overlap),(tamanho_quadro- tamanho_overlap)),(tamanho_quadro,1)).T
        indices = numpy.array(indices,dtype=numpy.int32)
        quadros = sinal_preenchido[indices]
        janela = numpy.tile(numpy.ones(tamanho_quadro),(num_quadros,1))
        quadros = quadros * janela

        #hamming window
        hamming = numpy.zeros(tamanho_quadro)
        for i in range(tamanho_quadro):
            hamming[i] = 0.54 - (0.46* numpy.cos(2*i*numpy.pi/(tamanho_quadro-1)))
        for i in range(0,len(quadros)):
            quadros[i] = numpy.multiply(quadros[i],hamming)

        self.bin_quadros = numpy.zeros((len(quadros),self.num_filtros_mel))

        for q in range(0, len(quadros)):
            #aplicar dft

            self.bin = fft(quadros[q])
            print(self.bin[0:5],' - ', self.bin[195:])
        return 1

    def grafico(self, matriz):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(numpy.transpose(matriz), interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()
#         bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
#         ax.text(9, 32, "Sample A", ha="center", va="center", size=20,bbox=bbox_props)
#         ax.text(20, 32, "Sample A", ha="center", va="center", size=20,bbox=bbox_props)
#         ax.text(40, 32, "Sample A", ha="center", va="center", size=20,bbox=bbox_props)
#         ax.text(60, 32, "Sample A", ha="center", va="center", size=20,bbox=bbox_props)

        plt.show()



#numpy.savetxt('coeficientes.txt', coeficientes, delimiter=',')

#(rate,sig) = wav.read("MeuProjeto_H_1_50_avance_2.wav")
#print(rate,'-',sig)

