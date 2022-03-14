### FUNÇÕES PARA EQUAÇÃO DE LAPLACE
import numpy as np
from scipy.linalg import solve as solve
import matplotlib.pyplot as plt


# import time


def malha(tam, dim, valor, neumann):
    '''
    Monta uma malha pronta para preenchimento com as condições de contorno
    :param tam: float, tamanho dos eixos da malha
    :param dim: int, numero de nós de cada eixo (em quantas partes o tamanho será dividido)
    :param valor: list, valores de temperatura/fluxo
    :param neumann: list, se for false é temperatura, true fluxo
    :return: T: Matriz de zeros de dimensões especificadas, dx: passo em X, dy: passo em Y.
    '''
    X = Y = np.linspace(0, float(tam), dim)  # Cria os eixos da matriz
    dx = X[1] - X[0]  # passo em x
    dy = Y[1] - Y[0]  # passo em y
    T = np.zeros([len(Y), len(X)])  # Cria uma matriz de zeros do tamanho dos eixos
    if neumann[0]:  # se for fluxo
        T = np.insert(T, len(T[:, 0]), -2 * dx * valor[0], axis=0)  # superior
    else:  # se for temperatura
        T[0, :] = valor[0]
    if neumann[1]:
        T = np.insert(T, 0, -2 * dx * valor[1], axis=0)  # inferior
    else:
        T[-1, :] = valor[1]
    if neumann[2]:
        T = np.insert(T, 0, -2 * dy * valor[2], axis=1)  # esquerdo
    else:
        T[:, 0] = valor[2]
    if neumann[3]:
        T = np.insert(T, len(T[0, :]), -2 * dy * valor[3], axis=1)  # direito
    else:
        T[:, -1] = valor[3]

    return T


def solucao_iterativa(T, Lim, neumann, dx, valor):
    '''Programa para solução iterativa da equação de Laplace.
    :param T: matriz com as condições de contorno
    :param Lim: limitante superior para o erro
    :param neumann: vetor com valores lógicos para fluxo
    :param dx: passo em x
    :param valor: vetor de temperaturas/fluxos dos contornos
    :return:
    T: matriz solucionada
    erro: matriz do erro relativo
    '''
    # aresta da malha (quadrada)
    # X = np.linspace(0, len(T), (len(T) * len(T[0])))
    y = len(T)  # eixo y
    x = len(T[0])  # eixo x
    lamb = 1.5  # lambda (sobrerrelaxação)

    # Auxiliares para o cálculo do erro relativo
    erro = np.zeros([y, x]) + 10  # matriz erro inicializada com 10 em todas as posições
    erro[0, :] = 0  # Se não forem definidos como 0, nunca será mudado e o loop (while) será infinito.
    erro[-1, :] = 0
    erro[:, -1] = 0
    erro[:, 0] = 0

    # Matriz auxiliar para T
    Taux = np.zeros([y, x])

    while np.any(erro > Lim):
        for i in range(1, y - 1):
            for j in range(1, x - 1):
                # -------- #  Ajuste dos pnts aux pra condições de fluxo # -------- #
                # Fluxo no Contorno Inferior
                if i == 1:
                    if neumann[1]:
                        T[0, j] = T[1, j] - 2 * dx * valor[1]

                # Fluxo no Contorno Esquerdo
                if j == 1:
                    if neumann[2]:
                        T[i, 0] = T[i, 1] - 2 * dx * valor[2]

                # ----------------------- # Pontos Internos # ------------------------ #
                Tnovo = (T[i - 1, j] + T[i + 1, j] + T[i, j - 1] + T[i, j + 1]) / 4
                T[i, j] = lamb * Tnovo + (1 - lamb) * T[i, j]

                # print(i,", ",j)
                # -------- #  Ajuste dos pnts aux finais pra condições de fluxo # -------- #
                # Fluxo no Contorno Superior
                if i == len(T[:, 0]) - 2:
                    # print("Ultima linha interna")
                    if neumann[0]:
                        # Temp no pnt aux
                        T[i + 1, j] = T[i - 1, j] - 2 * dx * valor[0]

                        # Recalculando a temp no contorno
                        Tnovo = (T[i - 1, j] + T[i + 1, j] + T[i, j - 1] + T[i, j + 1]) / 4
                        T[i, j] = lamb * Tnovo + (1 - lamb) * T[i, j]

                # Fluxo no Contorno Direito
                if j == len(T[0, :]) - 2:
                    # print("Ultima coluna interna")
                    if neumann[3]:
                        # Temp no pnt aux
                        T[i, j + 1] = T[i, j - 1] - 2 * dx * valor[3]

                        # Recalculando a temp no contorno
                        Tnovo = (T[i - 1, j] + T[i + 1, j] + T[i, j - 1] + T[i, j + 1]) / 4
                        T[i, j] = lamb * Tnovo + (1 - lamb) * T[i, j]

                erro[i, j] = abs((T[i, j] - Taux[i, j]) / T[i, j])  # Cálculo do erro relativo em cada posição
                # erro[i,j] = abs( (Tnovo - Taux[i,j])/Tnovo)
                Taux[i, j] = T[i, j]

    # RECORTES DOS PONTOS AUXILIARES
    if neumann[0]:
        T = T[:-1, :]  # SUPERIOR
    if neumann[1]:
        T = T[1:, :]  # INFERIOR
    if neumann[2]:
        T = T[:, 1:]  # ESQUERDO
    if neumann[3]:
        T = T[:, :-1]  # DIREITO

    return T


def solucao_sistemas(T, valor):
    '''
    Programa para solução por sistemas de equações da equação de Laplace
    :param T: array, malha formatada para ser resolvida
    :param valor: lista, temperaturas/fluxos dos contornos
    :return: array, malha com os valores internos preenchidos
    '''
    ##### VETOR DE TERMOS INDEPENDENTES #####
    n = len(T[0])
    C = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == 1:
                C[i][j] += T[i - 1][j]
            if i == n - 1:
                C[i - 1][j] += T[i][j]

            if j == 1:
                C[i][j] += T[i][j - 1]
            if j == n - 1:
                C[i][j - 1] += T[i][j]

    # Recorte de C
    K = C[1:n - 1, 1:n - 1]

    # Revel de -K: vetor dos termos independentes
    b = np.ravel(-K)

    ##### MATRIZ DE COEFICIENTES #####
    n_linhas = n  # número de linhas/colunas  na matriz das temperaturas
    N = n_linhas - 2  # número de linhas/colunas na matriz das incógnitas

    X = np.zeros([N ** 2, N ** 2])  # montagem da matriz das incógnitas
    Mult_N = np.array([N * i for i in np.arange(1, N ** 2 + 1, 1)])  # multiplos de N necessários para a condição
    # k = [N * i for i in np.arange(1, N ** 4, 1)]
    for i in range(N ** 2):
        for j in range(N ** 2):
            if i == j:
                X[i][j] = -4  # diagonal principal
                if j != 0:  # Coluna da esquerda: se a linha i não é a primeira
                    if (
                            j - 1) not in Mult_N - 1:  # Se a coluna da esquerda não é um a menos de um multiplo de N ela vale 1
                        X[i][j - 1] = 1
                if j - N >= 0:  # N-ésima coluna a partir da Diag. Princ.:
                    X[i][j - N] = 1
                if j != (N ** 2) - 1:
                    if (j + 1) % N != 0:
                        X[i][j + 1] = 1
                if j + N <= (
                        N ** 2) - 1:  # N-ésima coluna a partir da Diag. Princ.: se a coluna da direita não é a ultima, ela vale 1
                    X[i][j + N] = 1

    ##### SOLUÇÃO FINAL #####
    # Solução por solve()
    y = solve(X, b)

    # Tranformando y em uma matriz NxN
    y = y.reshape((N, N))

    # Expandido y para (N+2)x(N+2) com as codições de contorno
    # nas primeiras e ultimas linhas e colunas
    y = np.insert(y, n - 2, valor[0], axis=0)  # superior
    y = np.insert(y, 0, valor[1], axis=0)  # inferior
    y = np.insert(y, 0, valor[2], axis=1)  # esquerdo
    y = np.insert(y, n - 1, valor[3], axis=1)  # direito

    return y


def colorplot(T):
    '''
    Plota o gráfico 2D de cores da malha após a solução.
    :param T: matriz da malha solucionada
    '''
    # plots
    fig2D, axTemp = plt.subplots()  # Cria a figura com um subplot
    # pcolor(T, cmap='jet')
    plt.pcolormesh(T, cmap='jet')  # mair rápido que pcolor()
    # fig2D, (ax1,ax2) = plt.subplots(1,2) # Cria a figura com um subplot
    # pcolor(T, cmap='jet')
    cbar = plt.colorbar()

    axTemp.set_title('Temperatura', fontsize=16)
    cbar.ax.set_ylabel('Temperatura (°C)', fontsize=10)
    # fig2D.colorbar(axTemp, label='Temperatura (°C)')
    axTemp.set_xlabel('j', fontsize=14)
    axTemp.set_ylabel('i', fontsize=14)
    plt.show()
