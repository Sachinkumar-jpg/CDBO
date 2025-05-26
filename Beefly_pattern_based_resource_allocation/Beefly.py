import numpy as np
import random


def generate(r, c, l, u):
    Data=[]
    for row in range(r):
        tem = []
        for column in range(c):
            tem.append(random.uniform(l, u))
        Data.append(tem)
    return Data


def fitness1(soln):     # objective function
    F = []
    for i in range(len(soln)):
        F.append(random.random())
    return F


def fitness(N, P, P_max, throuhput):
    T = throuhput

    Fitness = [1/2 * ((1-T) + P)]
    Fitness = np.resize(Fitness, (len(N)))

    return Fitness


def capacity(Cmax, fit):
    pi = 3.14
    Capacity = []
    for i in range(len(Cmax)):
        ci = Cmax[i]*(np.sin((fit[i]-min(fit))/(max(fit)-min(fit)))*(pi/2))
        Capacity.append(ci)
    return Capacity


def movement(NB):

    mov=[]
    for i in range(len(NB)):
        for j in range(len(NB[i])):
            mov.append(np.sqrt(np.sum(np.square(NB[i][j]))))
    return mov


def Humidity(hum, di, pmax):

    itr = 1000
    p0 = 1
    Hum = []
    p = pmax-p0*(1-(itr/100)+random.uniform(0, 1))
    for i in range(len(hum)):
        for j in range(len(hum[i])):
            Hum.append(p*hum[i][j]*di[i])
    return Hum


def new(Hum, Nb_old):
    Nb_new=[]
    for i in range(len(Nb_old)):
        for j in range(len(Nb_old[i])):
            Nb_new.append(Nb_old[i][j]+Hum[i]*(Nb_old[i][j]))
    return Nb_new


def beefly(P_max, N_D2D, P_ini, throuhput):
    R, C, Lb, Ub = N_D2D, 5, 1, 5
    g, max_it = 0, 100
    soln = generate(R, C, Lb, Ub)

    while g < max_it:
        fit = fitness(soln, P_max, P_ini, throuhput)

        Mov = movement(soln)
        hum = Humidity(soln, Mov, P_max)
        Nb_new = new(hum, soln)
        bst = np.argmax(fit)
        best_soln = Nb_new[bst]
        g += 1

    bestX = abs(int(np.max(best_soln)))

    return bestX


