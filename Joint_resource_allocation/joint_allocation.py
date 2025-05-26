import math,random, numpy as np


def func(soln):
    Fit = []
    for i in range(len(soln)):
        F, hr = 0, random.random()
        for j in range(len(soln[i])):
            F += soln[i][j] * hr
        Fit.append(F)
    return Fit


def fitness(N, P, throuhput):

    T = throuhput

    Fitness = [1/2 * ((1-T) + P)]
    Fitness = np.resize(Fitness, len(N))

    return Fitness


def initialize(n, m):
    data = []
    for i in range(n):
        tem = []
        for j in range(m):
            tem.append(random.random())  # initial position
        data.append(tem)
    return data


def sort_by_fitness(fit, soln):
    fit = list(fit)
    sorted_soln = []
    fit_copy = fit.copy()
    for i in range(len(fit_copy)):
        index = fit_copy.index(np.min(fit_copy))
        fit_copy[index] = 100           # set by max. value to get next min.
        sorted_soln.append(soln[index])
    return sorted_soln


def split_FS(FS):
    FSht = []  # FS on hickory nut tree
    FSat = []  # FS on acorn nuts trees
    FSnt = []  # FS on normal trees
    O = len(FS) // 3            # split soln by 3
    for i in range(O):
        FSht.append(FS[i])
    for i in range(O, (O+O)):
        FSat.append(FS[i])
    for i in range((O+O), len(FS)):
        FSnt.append(FS[i])              # FS on normal tree
    return FSht, FSat, FSnt


def case_1(FS, Pdp, Gc, FSat, FSht):
    acorn = []
    for i in range(len(FSat)):
        R1, dg = random.random(), random.random()      # random number, random gliding distance
        if R1 >= Pdp:
            tem = []
            for j in range(len(FSat[i])):
                fsat = FSat[i][j] + (dg * Gc * (FSht[i][j] - FSat[i][j]))
                tem.append(fsat)
            acorn.append(tem)
        else :
            ss = int(random.random() * len(FS))
            acorn.append(FS[ss])
    return acorn


def case_2(FS, Pdp, Gc, FSnt, FSat):
    normal = []
    for i in range(len(FSnt)):
        R2,dg = random.random(), random.random()                     # random number, random gliding distance
        if R2 >= Pdp:
            tem = []
            for j in range(len(FSnt[i])):
                fsnt = FSnt[i][j] + (dg * Gc * (FSat[i][j] - FSnt[i][j]))
                tem.append(fsnt)
            normal.append(tem)
        else:
            ss = int(random.random() * len(FS))
            normal.append(FS[ss])
    return normal


def case_3(FS, Pdp, Gc, FSht, FSnt):
    hickory = []
    for i in range(len(FSnt)):
        R3, dg = random.random(), random.random()  # random number, random gliding distance
        if R3 >= Pdp:
            tem = []
            for j in range(len(FSnt[i])):
                fsnt = FSnt[i][j] + (dg * Gc * (FSht[i][j] - FSnt[i][j]))
                tem.append(fsnt)
            hickory.append(tem)
        else:
            ss = int(random.random() * len(FS))
            hickory.append(FS[ss])
    return hickory


def calc_seasonal_const(FSat, FSht):
    SC = []
    for i in range(len(FSat)):
        summ = 0
        for j in range(len(FSat[i])):
            summ += math.sqrt(math.pow((FSat[i][j] - FSht[i][j]), 2))
        SC.append(summ)
    return SC


def levy_fn():
    ra, rb, beta = random.random(), random.random(),  1.5
    sigma = math.pow(((1 + beta) * math.sin((3.14 * beta) / 2))
                     / (((1 + beta) / 2) * beta * math.pow(2, ((beta - 1) / 2))), (1 / beta))
    Levy = 0.01 * ((ra * sigma) / abs(math.pow(rb, (1 / beta))))
    return Levy


def flying_squirrel(FSht, FSat, FSnt):
    Fly_squ = []
    for i in range(len(FSht)):
        Fly_squ.append(FSht[i])
    for i in range(len(FSat)):
        Fly_squ.append(FSat[i])
    for i in range(len(FSnt)):
        Fly_squ.append(FSnt[i])
    return Fly_squ


def Algorithm(P_max, N_D2D, P_ini, throuhput):

    N, M, t, tm = N_D2D, 50, 0, 1000
    if N < 3:
        N = 3  # solution size must be > 3, since soln. is divided by 3
    else:
        n = N % 3  # make soln. as divisible of 3
        N = N - n

    FS = initialize(N, M)
    Fit = fitness(FS, P_ini, throuhput)
    FS = sort_by_fitness(Fit, FS)
    FSht, FSat, FSnt = split_FS(FS)

    Pdp, Gc = random.random(), 1.9          # Probability, gliding constant
    Overall_fit, Overall_best = [], []
    while t < tm:
        FSat = case_1(FS, Pdp, Gc, FSat, FSht)
        FSnt = case_2(FS, Pdp, Gc, FSnt, FSat)
        FSht = case_3(FS, Pdp, Gc, FSht, FSnt)
        Smin = (10E-6) / (math.pow(365, ((t + 1) / (tm / 2.5))))            # min.seasonal constant
        SC = calc_seasonal_const(FSat, FSht)
        for i in range(len(SC)):
            if (SC[i] < Smin):                  # satisfied Seasonal monitoring condition
                for j in range(len(FSnt[i])):
                    FSnt[i][j] = levy_fn()
        FS = flying_squirrel(FSht, FSat, FSnt)          # update FS
        Fit = func(FS)                  # Fitness Calculation
        Overall_fit.append(np.min(Fit))
        FS = sort_by_fitness(Fit, FS)       # sort the FS location by the ascending order(min) of Fitness
        Overall_best.append(FS[0])
        t += 1
    best = Overall_fit.index(np.min(Overall_fit))
    BEST_SOLUTION = np.mean(Overall_best[best])

    return BEST_SOLUTION





