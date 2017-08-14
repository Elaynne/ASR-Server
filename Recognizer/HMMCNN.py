'''
Created on 25/08/2014

@author: Rafael
'''
import numpy
import random
import dill
import theano
import theano.tensor as T
import pickle

class Hidden_Markov_Model:
    
    def __init__(self, transicoes = [],  estado_inicial = [], observacoes = [], rede = []):
        self.transicoes = transicoes
        self.estado_inicial = estado_inicial
        self.rede = rede

    def iniciar(self, transicoes):

        a_transicoes = numpy.random.rand(transicoes,transicoes)
        for i in range(0,transicoes):
            aux = {}
            for j in range(0,transicoes):
                aux[str(j+1)] = random.random()

        inicial = numpy.random.rand(transicoes)
        self.transicoes = a_transicoes
        self.estado_inicial = inicial
        with open('rede.pkl', 'rb') as f:
            self.rede = pickle.load(f)

    def calcular_probabilidades(self, entrada):
        batch_size = 100

        linhas = entrada.shape[0]
        index = T.lscalar()

        x = self.rede.salvation

        base = numpy.concatenate([entrada,numpy.zeros([batch_size - (linhas % batch_size),600])])
        rep_for = base.shape[0] / batch_size

        #base = numpy.tile(entrada[0,:],(batch_size,1))
        #for i in range(1,linhas):
        #    base = numpy.concatenate([base,numpy.tile(entrada[i,:],(batch_size,1))])

        base = theano.shared(numpy.asarray(base,dtype="float64"),borrow=True)
        theano.config.exception_verbosity = 'high'
        #theano.printing.debugprint(self.rede.p_y_given_x)

        predicao = theano.function([index], self.rede.p_y_given_x,
         givens={\
            x: base[index * batch_size: (index + 1) * batch_size]}, on_unused_input='ignore')

        probabilidades = predicao(0).reshape(batch_size,16)

        for i in range(1,int(rep_for)):
            aux =  predicao(i)
            probabilidades = numpy.concatenate([probabilidades, aux.reshape(batch_size,16)])

        return probabilidades[0:linhas,:]

    def forward(self, obs, prob = []):
        if len(prob) == 0:
            observacoes = self.calcular_probabilidades(obs)
        else:
            observacoes = prob
        alpha = numpy.zeros((len(obs),len(self.transicoes)))
        
        #Inicializacao
        for c in range(0,len(self.transicoes)):
            alpha[0][c] = self.estado_inicial[c] * observacoes[0][c]
            
        
        #Inducao
        for c_obs in range(1,len(obs)):            
            for c_tr in range(0,len(self.transicoes)):
                alpha[c_obs][c_tr] = numpy.dot(alpha[c_obs-1],numpy.transpose(self.transicoes)[c_tr])
                alpha[c_obs][c_tr] *= observacoes[c_obs][c_tr]
                
                
        #Terminacao
        return alpha, numpy.sum(alpha[len(obs) - 1])
        
        
    def backward(self, obs, prob = []):
        if len(prob) == 0:
            observacoes = self.calcular_probabilidades(obs)
        else:
            observacoes = prob
        beta = numpy.zeros((len(obs),len(self.transicoes)))
        
        #Inicializacao
        for c in range(0,len(self.transicoes)):
            beta[len(obs)-1][c] = 1
        
        #Inducao
        for c_obs in range(len(obs)-2,-1,-1):
            for c_tr in range(0,len(self.transicoes)):
                soma = 0
                for c_col in range(0,len(self.transicoes)):
                    soma += self.transicoes[c_tr][c_col] * observacoes[c_obs+1][c_col] * beta[c_obs+1][c_col]
                                
                beta[c_obs][c_tr] = soma
        
        #Terminacao
        soma = 0
        
        for c in range(0,len(self.transicoes)):
            soma += self.estado_inicial[c] * observacoes[0][c] * beta[0][c]
        
        return beta, soma
            
    def viterbi(self, obs, prob = []):
        if len(prob) == 0:
            observacoes = self.calcular_probabilidades(obs)
        else:
            observacoes = prob


        delta = numpy.zeros((len(obs),len(self.transicoes)))
        psi = numpy.zeros((len(obs),len(self.transicoes)))
                
        #Inicializacao
        for c in range(0,len(self.transicoes)):
            delta[0][c] = self.estado_inicial[c] * observacoes[0][c]
            
        #Recursao
        for c_obs in range(1,len(obs)):
            for c_tr in range(0,len(self.transicoes)):
                maior, arg_maior = 0, 0
                for c_col in range(0,len(self.transicoes)):
                    atual = delta[c_obs-1][c_col] * self.transicoes[c_col][c_tr]
                    if atual > maior:
                        arg_maior = c_col
                        maior = atual
                delta[c_obs][c_tr] = maior
                psi[c_obs][c_tr] = arg_maior
                  
                delta[c_obs][c_tr] *= observacoes[c_obs][c_tr]
                
        
        #Terminacao
        maior, arg_maior = 0, 0
        for c in range(0,len(self.transicoes)):
            if delta[len(obs)-1][c] > maior:
                arg_maior = c
                maior = delta[len(obs)-1][c]
        
        melhor_seq = numpy.zeros(len(obs))
        melhor_seq[len(obs)-1] = arg_maior
        for c in range(len(obs)-1,0,-1):
            melhor_seq[c-1] = psi[c][melhor_seq[c]]        
        
        return maior, melhor_seq
    
    def p(self, t, i, j, obs, alpha, beta, prob = []):

        if len(prob) == 0:
            observacoes = self.calcular_probabilidades(obs)
        else:
            observacoes = prob

        numerador = 0
        if t == (len(obs) - 1):
            numerador = alpha[t][i] * self.transicoes[i][j]
        else:
            numerador = alpha[t][i] * self.transicoes[i][j] * observacoes[t+1][j] * beta[t+1][j]
        
        denominador = 0
        
        for c in range(len(self.transicoes)):
            denominador += (alpha[t][c] * beta[t][c])
       
#         
#         if denominador == 0 :
#             return 1
        return numerador / denominador
    
    def gamma(self, i, t, obs, alpha, beta):
        numerador = alpha[t][i] * beta[t][i]
        denominador = 0
         
        for c in range(len(self.transicoes)):
            denominador += (alpha[t][c] * beta[t][c])
        
#         if denominador == 0 :
#             return 1
        return (numerador / denominador)
    
    def treinar(self, obs, etapas, imprimir = False):

        sequencia = len(obs)
        transicoes1 = numpy.zeros((len(self.transicoes),len(self.transicoes)))
        estado_inicial1 = numpy.zeros(len(self.transicoes))
        print('wow')
        for s in range(etapas):
            observacoes = self.calcular_probabilidades(obs)
            alpha, prob = self.forward(obs,observacoes)
            beta = self.backward(obs,observacoes)[0]
            print('etapa',s,prob)
            
            #Reestimar probabilidades iniciais
            for i in range(len(self.transicoes)):
                estado_inicial1[i] = self.gamma(i, 0, obs, alpha, beta)
                #print('i', estado_inicial1[i])

            #Reestimar probabilidades das transicoes
            for i in range(len(self.transicoes)):
                for j in range(len(self.transicoes)):
                    numerador = 0
                    denominador = 0
                    for t in range(sequencia-1):
                        numerador += self.p(t, i, j, obs, alpha, beta,observacoes)
                        denominador += self.gamma(i, t, obs, alpha, beta)
                    
                    transicoes1[i][j] = numerador / denominador
                    #print('t', i, j, transicoes1[i][j])
                


            self.estado_inicial = estado_inicial1
            self.transicoes = transicoes1

            

            
            if (imprimir):
                print('etapa final', s)
                print('estado_inicial', self.estado_inicial)
                print('transicoes',self.transicoes)

    
    

def randomizar_amostra(quant, porcentagem, face_viciada, outra_face):
    lista = []
    for c in range(quant):
        if (random.random() < porcentagem):
            lista.append(face_viciada)
        else:
            lista.append(outra_face)       
    return ''.join(lista)

def modelo_viciado(quant):
    transicao12 = 0.75
    transicao21 = 0.4
    emissao_h1 = 0.5
    emissao_h2 = 0.8
    
    lista = []
    emissao = emissao_h1
    transicao = transicao12
    for c in range(quant):
        if (random.random() < emissao):
            lista.append('h')
        else:
            lista.append('t')
        
        if (random.random() < transicao):
            transicao = transicao12 if transicao == transicao21 else transicao21
            emissao = emissao_h1 if transicao == transicao12 else emissao_h2
    
    return ''.join(lista)                 
    
# def main():
#     
#     #transicoes = [[0.5,0.5],[0.2,0.8]]
#     #observacoes = [{'h':0.2,'t':0.8},{'h':0.7,'t':0.3}]
#     #inicial = [0.6,0.4,]
#     #transicoes = [[0.3,0.5,0.2],[0,0.3,0.7],[0,0,1]]
#     #observacoes = [{'a': 1,'b': 0},{'a': 0.5,'b': 0.5},{'a': 0,'b': 1}]
#     #inicial = [0.6,0.4,0]
#     
#     transicoes = [[0.5,0.5],[0.5,0.5]]
#     observacoes = [{'h':0.5,'t':0.5,},{'h':0.5,'t':0.5}]
#     inicial = [0.49,.51]
#     #transicoes = [[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],]
#     #observacoes = [{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4}]
#     #inicial = [0.2,0.2,0.2,0.2,0.2]
#     modelo1 = Hidden_Markov_Model(transicoes, observacoes, inicial)
#         
#     transicoes = [[0.5,0.5],[0.5,0.5]]
#     observacoes = [{'h':0.5,'t':0.5,},{'h':0.5,'t':0.5}]
#     inicial = [0.50,0.50]
#     #transicoes = [[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],]
#     #observacoes = [{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4},{'h':0.3,'t':0.3,'e':0.4}]
#     #inicial = [0.2,0.2,0.2,0.2,0.2]
#     modelo2 = Hidden_Markov_Model(transicoes, observacoes, inicial)
#     
#     #seq1 = randomizar_amostra(1000,0.9,'h','t')
#     seq2 = randomizar_amostra(1000,0.5,'h','t')
#     
#     #print(seq1)
#     #print(seq2)
#     
#     
#     
#     
#     etapas = 1000
#     #vicio h
#     for c in range(50):
#         modelo1.treinar(modelo_viciado(200), etapas)
#         
#     #equilibrado
#     if False:
#         modelo2.treinar(seq2, etapas)
#         print("probabilidade modelo viciado", modelo1.forward(modelo_viciado(20))[1])
#         obs1 = modelo_viciado(200)
#         print('probabilidade no viciado', '%0.200f' % modelo1.forward(obs1)[1])
#         print('probabilidade no equilibrado', '%0.200f' % modelo2.forward(obs1)[1])
#     
#     if True:
#         print(modelo1.transicoes)
#         print(modelo1.estado_inicial)
#         print(modelo1.observacoes)
#         print(modelo1.viterbi(obs1))
#         print('-------')
#         print(modelo2.transicoes)
#         print(modelo2.estado_inicial)
#         print(modelo2.observacoes)
#         print(modelo2.viterbi(obs1))
#         print(len(set('abcbabcabcbacbab')))
#     
# if __name__ == "__main__": main()