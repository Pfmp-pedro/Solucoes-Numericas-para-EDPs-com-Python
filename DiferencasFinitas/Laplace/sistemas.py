import numpy as np
from scipy.linalg import solve as solve
import time
#import solve


def termos_ind (T):
    '''Função que monta o vetor 'b' dos termos independentes 
    a partir da matriz 'T' com as condições de contorno'''
    n = len(T[0])
    C = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == 1:
                C[i][j] += T[i-1][j]
            if i == n-1: 
                C[i-1][j] += T[i][j]

            if j == 1:
                C[i][j] += T[i][j-1]
            if j == n-1: 
                C[i][j-1] += T[i][j]
     
    # Recorte de C
    K = C[1:n-1,1:n-1]

    # Revel de -K: vetor dos termos independentes
    b = np.ravel(-K)
    return b



def coef_T (n):
    '''Esta função monta a matriz dos coeficientes de um 
    sistema de equações de diferenças problemas do tipo:
    laplaciano de T = 0, cond. de cont. com temperatura constante, malha quadrada.
    Parâmetro: n = número de linha na malha'''
    n_linhas = n                       # número de linhas/colunas  na matriz das temperaturas
    N = n_linhas-2                     # número de linhas/colunas na matriz das incógnitas

    X = np.zeros([N**2,N**2])          # montagem da matriz das incógnitas
    Mult_N = np.array([N*i for i in np.arange(1, N**2 +1, 1)])
    k = [N*i for i in np.arange(1, N**4, 1)]
    for i in range(N**2): #[OK]
        for j in range(N**2): #[OK]
            if i == j:
                aux = int(N*i)
                #print(Mult_N+1)
                X[i][j] = -4               # Diagonal Principal
                if i < N**2-1:             # Coluna da direita: se a linha i não é a ultima, o vizinho da direita vale 1
                    if j+1 not in Mult_N:  # Se a coluna da direita não é um multiplo de N ela vale 1
                        X[i][j+1] = 1
                if i > 0:                  # Coluna da esquerda: se a linha i não é a primeira
                    if j-1 not in Mult_N-1: # Se a coluna da esquerda não é um a menos de um multiplo de N ela vale 1
                        X[i][j-1] = 1 
                if j+N <= N**2-1:          # N-ésima coluna a partir da Diag. Princ.: se a coluna da direita não é a ultima, ela vale 1
                    X[i][j+N] = 1  
                if j-N >= 0:               # N-ésima coluna a partir da Diag. Princ.:
                    X[i][j-N] = 1
    return X

def solucao_solve(A, b, n, x_e, x_d, x_s, x_i):
    '''Parâmetros: 
    A = matriz dos coeficientes
    b = vetor dos t. independentes
    n = número de linhas da malha
    x_e = temperatura no lado esquerdo
    x_d = // lado direito
    x_s = // lado superior
    x_i = // lado inferior
  
    Retorno: matriz y
    '''

    N = n-2

    # Solução por solve()
    y = solve(A,b)

    # Tranformando y em uma matriz NxN
    y = y.reshape((N,N))

    # Expandido y para (N+2)x(N+2) com as codições de contorno
    # nas primeiras e ultimas linhas e colunas
    y = np.insert(y, 0, x_e, axis=1)
    y = np.insert(y, n-1, x_d, axis=1)
    y = np.insert(y, n-2, x_s, axis=0)
    y = np.insert(y, 0, x_i, axis=0)

    return y

def solucao_iterativa(T, Lim):
    '''Programa para solução iterativa. 
    Parâmetros: 
    T: matriz com as condições de contorno
    Lim: limitante superior para o erro
    Retorno:
    T: matriz solucionada
    erro: matriz do erro relativo
    '''
    # aresta da malha (quadrada)
    n = len(T[0]) 
    lamb = 1.5
    
    # Auxiliares para o cálculo do erro relativo
    erro = np.zeros([n, n]) + 10 # matriz erro inicializada com 10 em todas as posições
    erro[0,  :] = 0 # Se não forem definidos como 0, nunca será mudado e o loop (while) será infinito.
    erro[-1, :] = 0
    erro[:, -1] = 0
    erro[:,  0] = 0

    # Matriz auxiliar para T
    Taux = np.zeros([n, n])

    # Auxiliares para contagem de iterações
    iterac = 0
    loops = 0

    # Auxiliares para a contagem do tempo
    t1 = time.clock()

    while np.any(erro > Lim):
        for i in range(1,n-1):
            for j in range(1,n-1):
                Tnovo = (T[i-1,j]+T[i+1,j]+T[i,j-1]+T[i,j+1])/4
                T[i,j]=lamb*Tnovo+(1-lamb)*T[i,j]

                erro[i,j] = abs( (T[i,j] - Taux[i,j])/T[i,j]) # Cálculo do erro relativo em cada posição
                Taux[i,j] = T[i,j]
                loops += 1
        iterac += 1

    t2 = time.clock()

    print("         Dimensão da malha:", n, "linhas/colunas")
    print("   Limite superior do erro:", Lim*100, "%")
    print ("                 Iterações:", iterac)
    print ("           Pontos iterados:", loops)
    print(" Tempo de Iteração (clock):", np.round(t2-t1,3), "segundos")
    
    return T, erro





