import numpy as np
import warnings
warnings.filterwarnings("ignore")
import math


def fitness(N, P, P_max, throuhput):
    # P = Power allocated to D2D Pair
    # P_max = Maximum allowable power of pair

    # T = throughput of D2D pair
    T = throuhput

    Fitness = [1/2 * ((1-T) + P)]
    Fitness = np.resize(Fitness, (N))

    return Fitness


class funtion():
    def __init__(self):
        print("starting Optimization")


def Parameters():
    fobj = F1
    lb = -10
    ub = 10
    dim = 30
    return fobj, lb, ub, dim


def F1(x):
    o = np.sum(np.square(x))
    return o


def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp


def Boundss(ss, LLb, UUb):
    temp = ss
    for i in range(len(ss)):
        if temp[i] < LLb[0, i]:
            temp[i] = LLb[0, i]
        elif temp[i] > UUb[0, i]:
            temp[i] = UUb[0, i]
    return temp


def swapfun(ss):
    temp = ss
    o = np.zeros((1, len(temp)))
    for i in range(len(ss)):
        o[0, i] = temp[i]
    return o


def DBO(pop, M, c, d, dim, fun, throughput, P_i, P_max):

    P_percent = 0.2
    pop = pop + 20
    pNum = round(pop * P_percent)
    lb = c * np.ones((1, dim))
    ub = d * np.ones((1, dim))
    X = np.zeros((pop, dim))

    # fit = np.zeros((pop, 1))
    fit = fitness(pop, P_i, P_max, throughput)              # Fitness
    fit = fit.reshape(-1, 1)

    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        fit[i] = fun(X[i, :])
    pFit = fit
    pX = X
    XX = pX
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]
    Convergence_curve = np.zeros((1, M))

    for t in range(M):
        B = np.argmax(pFit[:, 0])
        worse = X[B, :]  #
        r2 = np.random.rand(1)
        for i in range(pNum):
            if r2 < 0.9:
                a = np.random.rand(1)
                if a > 0.1:
                    a = 1
                else:
                    a = -1
                X[i, :] = pX[i, :] + 0.3 * np.abs(pX[i, :] - worse) + a * 0.1 * (XX[i, :])      # Equation(1)
            else:
                aaa = np.random.randint(180, size=1)
                if aaa == 0 or aaa == 90 or aaa == 180:
                    X[i, :] = pX[i, :]
                theta = aaa * math.pi / 180
                # X[i, :] = pX[i, :] + math.tan(theta) * np.abs(pX[i, :] - XX[i, :])              # Equation(2)

                del0, del1, del2 = 0, 0.5, 0.6
                # -------------- Equation Update --------------
                X[i, :] = (del0 + (del1*pX[i, :]) + (del2*pX[2, :]) + (del1*fun(X[i, :])) + (del2*fun(X[2, :]))) * \
                          (1+math.tan(theta)) - (math.tan(theta) * pX[i, :])

            X[i, :] = Bounds(X[i, :], lb, ub)
            fit[i, 0] = fun(X[i, :])

        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :]

        R = 1 - t / M
        Xnew1 = bestXX * (1 - R)
        Xnew2 = bestXX * (1 + R)
        Xnew1 = Bounds(Xnew1, lb, ub)               # Equation(3)
        Xnew2 = Bounds(Xnew2, lb, ub)
        Xnew11 = bestX * (1 - R)
        Xnew22 = bestX * (1 + R)                    # Equation(5)
        Xnew11 = Bounds(Xnew11, lb, ub)
        Xnew22 = Bounds(Xnew22, lb, ub)
        xLB = swapfun(Xnew1)
        xUB = swapfun(Xnew2)

        for i in range(pNum + 1, 12):               # Equation(4)
            X[i, :] = bestXX + (np.random.rand(1, dim)) * (pX[i, :] - Xnew1) + (np.random.rand(1, dim)) * (
                        pX[i, :] - Xnew2)
            X[i, :] = Bounds(X[i, :], xLB, xUB)
            fit[i, 0] = fun(X[i, :])
        for i in range(13, 19):                     # Equation(6)
            X[i, :] = pX[i, :] + (
                        (np.random.randn(1)) * (pX[i, :] - Xnew11) + ((np.random.rand(1, dim)) * (pX[i, :] - Xnew22)))
            X[i, :] = Bounds(X[i, :], lb, ub)
            fit[i, 0] = fun(X[i, :])
        for j in range(20, pop):                    # Equation(7)
            X[j, :] = bestX + np.random.randn(1, dim) * (np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :] - bestX)) / 2
            X[j, :] = Bounds(X[j, :], lb, ub)
            fit[j, 0] = fun(X[j, :])

        # Update the individual's best fitness value and the global best fitness value
        XX = pX
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]

        Convergence_curve[0, t] = fMin

    return fMin, bestX, Convergence_curve


def caviar_DBO(P_max, N_D2D, P_ini, throuhput):
    SearchAgents_no = N_D2D
    Max_iteration = 100

    fobj, lb, ub, dim = Parameters()
    fMin, bestX, DBO_curve = DBO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj, throuhput, P_ini, P_max)

    bestX = abs(int(np.max(bestX)))

    return bestX
