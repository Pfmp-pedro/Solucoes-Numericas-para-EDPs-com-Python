import numpy as np
import matplotlib.pyplot as plt


def malha(L, dx, tf, dt, valor, fluxo):
    '''
        Monta uma malha pronta para preenchimento com as condições de contorno
        :param L: float, comprimento da barra
        :param dx: float, passo de distância (em quantas partes o tamanho será dividido)
        :param tf: float, instante final da iteração
        :param dt: float, passo de tempo em função de lambda
        :param valor: list, valores de temperatura/fluxo do lado esquerdo e direito
        :param fluxo: list, se for false é temperatura, true fluxo
        :return: T: Matriz de zeros de dimensões especificadas.
        '''
    # ----------------------------------------------------------------- #
    #                     INICIALIZAÇÃO DOS VETORES                     #
    # ----------------------------------------------------------------- #
    X = np.arange(0, L + dx, dx)  # Vetor da coordenada X
    Inst = np.arange(0, tf + dt, dt)  # Vetor dos instantes t
    # fluxo = np.zeros(2)  # Vetor das flags de cond. de fluxo

    # --- Inicialização da Matriz de Temperaturas
    # primeira posição: instante, segunda: coordenada 'x'
    T = np.zeros([len(Inst), len(X)])  # T[instante, posição]

    # ----------------------------------------------------------------- #
    #                          FLAGS DE FLUXO                           #
    # ----------------------------------------------------------------- #
    # Letras = 'd D f F'  # String das letras que indicam cond. de fluxo
    # letras = Letras.split()  # Criando um vetor das letras separadas

    # ----------------------------------------------------------------- #
    #                         CONDIÇÃO INICIAL                          #
    # ----------------------------------------------------------------- #
    # Caso a barra estaja pré-aquecida inicializar o vetor com a(s) temperatura(s) inicial(ais)
    # for i in range(1,len(X) -1):
    #    T[0,i] = 2*X[i]

    # ----------------------------------------------------------------- #
    #                       CONDIÇÕES DE CONTORNO                       #
    # ----------------------------------------------------------------- #
    # Se for digitada uma das letras 'd' ou 'f' é interpretada uma cond.
    # de fluxo/derivada. Nesses casos insere-se uma nova coluna na primeira
    # ou ultima posição da matriz e a preenche com o valor auxiliar .
    # ----------------------------------------------------------------- #

    # --- Lado Esquerdo
    T_esq = valor[0]  # input("Temperatura no lado esquerdo (digite d para cond. na derivada): ")
    if fluxo[0]:
        # T_esq = float(input("dT/dx no lado esquerdo: "))
        dT = T[0, 1] - 2 * dx * T_esq  # Valor inicial da coluna auxiliar para fixar a derivada
        T = np.insert(T, 0, dT, axis=1)  # Insere o valor 'dT' na coluna (axis=1) 0 do vetor 'T'
        # fluxo[0] = 1  # Flag de condição no fluxo
    else:
        # Condição de Temperatura no Lado Esquerdo
        T[:, 0] = float(T_esq)

    # --- Lado Direito
    T_dir = valor[1]  # input(" Temperatura no lado direito (digite d para cond. na derivada): ")
    if fluxo[1]:
        # T_dir = float(input("dT/dx no lado direito: "))
        # Valor da coluna auxiliar para fixar a derivada
        dT = T[0, -1] + 2 * dx * T_dir
        # Insere uma nova coluna na ultima posição da matriz e a preenche com dT
        T = np.insert(T, len(T[0]), dT, axis=1)
        # Flag de condição no fluxo
        # fluxo[1] = 1
    else:
        # Condição de Temperatura no Lado Direito
        T[:, -1] = float(T_dir)

    # Impressão do vetor da condição inicial
    # print("T inicial = ",T[0,:])
    return T


def solucao(T, dx, lamb, valor, fluxo):
    '''
            Monta uma malha pronta para preenchimento com as condições de contorno
            :param T: array, matriz de zeros da malha
            :param dx: float, passo de distância (em quantas partes o tamanho será dividido)
            :param lamb: float, lambda, valor que garante convergencia e estabilidade
            :param valor: list, valores de temperatura/fluxo do lado esquerdo e direito
            :param fluxo: list, se for false é temperatura, true fluxo
            :return: T: Matriz da malha com os valores pós solução iterativa.
            '''
    # ----------------------------------------------------------------- #
    #                              SOLUÇÃO                              #
    # ----------------------------------------------------------------- #
    nt = len(T[:, 0])  # Número de instantes, incluindo as posições de fluxo
    nx = len(T[0, :])  # Número de coordenadas, incluindo as posições de fluxo

    for t in range(1,
                   nt):  # Percorre cada instante 't' ignorando a posição [0], o tempo não possui condição limitante (aberto no infinito)
        for x in range(1,
                       nx - 1):  # Percorre cada coordenada 'x' ignorando a posição [0], a coordenada 'x' é limitada pela CC no lado direito, por isso 'nx-1'
            # --- Caso cond. de fluxo no lado esquerdo
            if fluxo[0]:
                T[t, 0] = T[t, 2] - 2 * dx * valor[0]

            # --- Cálculo das temperaturas no pontos internos
            T[t, x] = T[t - 1, x] + lamb * (T[t - 1, x + 1] - 2 * T[t - 1, x] + T[t - 1, x - 1])

            # --- Caso cond. fluxo no lado direito
            # (A ultima temperatura é calculada depois da antepenultima,
            # assim o fluxo do lado direito deve vir após o calculo das
            # temperaturas ao longo da barra)
            if fluxo[1]:
                T[t, -1] = T[t, -3] + 2 * dx * valor[1]

    # Impressão das temperaturas em cada instante
    # print('T = ',np.round(T[t,:],2))

    # --- Recorte das posições auxiliares das cond. de fluxo
    if fluxo[0]:  # lado esquerdo
        T = T[:, 1:]
    if fluxo[1]:  # lado direito
        T = T[:, :-1]

    # Impressão da matriz de temperaturas
    # print("T_0 = ", T)
    return T


# ----------------------------------------------------------------- #
#                               PLOTS                               #
# ----------------------------------------------------------------- #

def colorplot(L, T, tf):
    '''
        Plota o gráfico 2D da malha após a solução.
        :param L: float, comprimento da barra
        :param T: matriz da malha solucionada
        :param tf: float, instante final da iteração
    '''
    # --- MAPA DE CORES
    fig, ax = plt.subplots()  # Cria a figura com um subplot
    xPlotCor = np.linspace(0, L,
                           len(T[0,
                               :]) + 1)  # Eixo x para o plot, sem 0 +1 não será mostrada a ultima coluna (direita)
    tPlotCor = np.linspace(0, tf, len(T[:, 0]))  # Eixo y para o plot
    plt.pcolor(xPlotCor, tPlotCor, T, cmap='jet')  # Mapa de cores com eixos rotulados certos
    # plt.pcolor(T, cmap='jet')                    # Mapa de cores com rótulos errados, indicando as 'posições' 'i' e 'j'
    cbar = plt.colorbar()  # Criando objeto barra de cores (para por rótulo)

    # -- Rótulos
    cbar.ax.set_ylabel('Temperatura (°C)', fontsize=16)  # Rótulo da barra de cores
    ax.set_ylabel('Tempo (s)', fontsize=16)  # Rótulo do eixo vertical
    ax.set_xlabel('Comprimento (cm)', fontsize=16)  # Rótulo do eixo vertical

    # -- Mostrar ou Salvar Figura
    plt.show()
    # savefig('Calor_Barra_-_Cores.pdf')


def lineplot(X, T, nt, Inst):
    # --- CURVAS PARA INSTANTES DISTINTOS
    fig, ax = plt.subplots()  # Cria a figura com um subplot
    plt.plot(X, T[nt // 5, :],  # 1/5 de tf
             X, T[2 * nt // 5, :],  # 2/5 de tf
             X, T[3 * nt // 5, :],  # 3/5 de tf
             X, T[4 * nt // 5, :],  # 4/5 de tf
             X, T[nt - 1, :])  # Instante Final tf

    # -- Legendas
    plt.legend(['t = ' + str(np.round(Inst[nt // 5], 3)) + ' s',  # 1/5 de tf
                't = ' + str(np.round(Inst[2 * nt // 5], 3)) + ' s',  # 2/5 de tf
                't = ' + str(np.round(Inst[3 * nt // 5], 3)) + ' s',  # 3/5 de tf
                't = ' + str(np.round(Inst[4 * nt // 5], 3)) + ' s',  # 4/5 de tf
                't = ' + str(np.round(Inst[nt - 1], 3)) + ' s'])  # Instante Final tf

    # -- Rótulos
    plt.xlabel('x [cm]', fontsize=16)
    plt.ylabel('Temperatura [°C]', fontsize=16)

    # -- Mostrar ou Salvar Figura
    plt.show()
    # savefig('Calor_Barra_-_Curvas.pdf')
