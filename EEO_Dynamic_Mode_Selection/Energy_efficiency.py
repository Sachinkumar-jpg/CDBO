import random, numpy as np


def Algorithm(P_max, N_D2D, P_ini, throuhput):

    def fitness1(N, P, throuhput):
        T = throuhput

        Fitness = [1 / 2 * ((1 - T) + P)]
        Fitness = np.resize(Fitness, (len(N)))

        return Fitness

    Local_Limit_Count = 0

    def initialize_soln(N, D):
        soln = []
        for i in range(N):
            tem = []
            for j in range(D):
                tem.append(random.random())
            soln.append(tem)
        return soln

    def check(i, N):
        r = random.randint(1, N)                # select random solution
        while i == r:                         # if r == i, change r
            r = random.randint(1, N)
        return r

    def fitness(soln):
        fit = []
        for i in range(len(soln)):
            summ = 0
            for j in range(len(soln[i])):
                summ += soln[i][j]
            fit.append(1/summ)
        return fit

    def global_leader_phase(GL, soln, prob):
        pos_update = []
        for i in range(len(soln)):
            tem = []
            r = check(i, len(soln)-1)
            U = random.random()
            U1 = random.random() * (1 - (-1)) - 1           # U(0, 1), U1(-1, 1)
            for j in range(len(soln[i])):
                if U < prob[i]:
                    SM_new = soln[i][j] + (U * (GL[j] - soln[i][j]))+ (U1 * (soln[r][j] - soln[i][j]))
                    tem.append(abs(SM_new))
                else:
                    tem.append(soln[i][j])
            pos_update.append(tem)
        return pos_update

    def calc_probability(Fit):
        prob = []
        max_fit = np.max(Fit)
        for i in range(len(Fit)):
            p = 0.9*(Fit[i]/max_fit)+0.1
            prob.append(abs(p))
        return prob

    def local_leader_phase(LL, solution, pr):
        pos_update = []
        for i in range(len(solution)):
            tem = []
            r = check(i, len(solution)-1)                   # random soln, where r!= i
            U,U1  = random.random(), (random.random()) * (1 - (-1)) - 1              # U(0, 1), U1(-1, 1)
            for j in range(len(solution[i])):
                if U >= pr:
                    SM_new = solution[i][j] + (U * (LL[j] - solution[i][j])) + (U1 * (solution[r][j] - solution[i][j]))
                    tem.append(abs(SM_new))
                else:
                    tem.append(solution[i][j])
            pos_update.append(tem)
        return pos_update

    def find_global_leader(overall_fit, overall_best, GL, Global_Limit_Count):
        b_in = overall_fit.index(np.min(overall_fit))           # best fitness index
        g_leader = overall_best[b_in]                           # soln in that best fitness index is global leader
        if GL == g_leader:
            Global_Limit_Count += 1                             # if not updated, increment Global_Limit_Count
        return g_leader

    def local_leader_decision_phase(solution, pr, GL, LL):
        pos_update = []
        SM_min, SM_max = 0, 1
        for i in range(len(solution)):
            tem = []
            U = random.random()             # U(0, 1)
            for j in range(len(solution[i])):
                if U >= pr:
                    SM_new = SM_min+(U*(SM_max-SM_min))
                    tem.append(abs(SM_new))
                else:
                    SM_new = solution[i][j] + (U * (GL[j] - solution[i][j]))+ (U * (solution[i][j] - LL[j]))
                    tem.append(abs(SM_new))
            pos_update.append(tem)
        return pos_update

    def global_leader_decision_phase(solution, MG, n_group):
        pos_update = []
        if n_group < MG:
            for j in range(len(solution)//2):
                pos_update.append(solution[j])
        return pos_update

    overall_fit = []
    overall_best = []
    # step 1: Initialize Population, Local Leader Limit, Global Leader Limit, pr.
    ra = random.random()
    N, D, max_itr, k = N_D2D, P_max, 100, 0              # row size, column size, max. iteration
    MG = N / 10                                 # max. group
    GL_limit = random.randint(1, (((2 * N) - (N / 2) + 1) + (N / 2)))
    LL_limit = D * N                            # global leader limit [N/2, 2×N]
    pr = random.random() * (0.9 - 0.1) + 0.1    # perturbation rate [0.1, 0.9]
    solution = initialize_soln(N, D)
    # step 2: Calculate ﬁtness
    Fit = fitness(solution)
    best_fit = np.min(Fit)
    overall_fit.append(best_fit)  # pop best fit
    b_index = np.argmin(Fit)        # index of best fit
    b_soln = solution[b_index]   # current best soln.
    overall_best.append(b_soln)   # pop best soln.
    # step 3: Select global leader and local leaders by applying greedy selection
    GL = overall_best[0]            # initial best
    LL = b_soln

    while k < max_itr:
        existing_position = solution.copy()
        Fit = fitness1(existing_position, P_ini, throuhput)
        best_fit = np.min(Fit)          # best fitness
        #  /* 1. Position update in Local Leader Phase */
        new_position = local_leader_phase(LL, solution, pr)
        new_fit = fitness(new_position)         # fitness calc of new position
        # /* 2. check best fitness */
        if np.min(new_fit) < best_fit:
            best_fit = np.min(new_fit)      # if new position have better fitness, update best soln
            overall_fit.append(best_fit)  # pop best fit
            Fit = new_fit  # update Fit
            b_soln = new_position[new_fit.index(best_fit)]
            overall_best.append(b_soln)
            solution = new_position.copy()      # update the soln.
        else:
            Fit = list(Fit)
            b_soln = existing_position[Fit.index(best_fit)]
        # /* 3. Calculate Probability */
        prob = calc_probability(Fit)
        # /* 4. Position update in Global Leader Phase */
        new_position = global_leader_phase(GL, solution, prob)
        solution = new_position.copy()
        # /* 5. update Global & Local leader */
        Global_Limit_Count = 0
        GL = find_global_leader(overall_fit, overall_best, GL, Global_Limit_Count)
        if LL == b_soln:
            Local_Limit_Count += 1          # if not updated, increment Local_Limit_Count
            LL = b_soln.copy()
        # /* 6. Local leader decision phase */
        if Local_Limit_Count > LL_limit:
            Local_Limit_Count = 0
            new_position = local_leader_decision_phase(solution, pr, GL, LL)
        solution = new_position.copy()      # update solution
        # /* 7. Global leader decision phase */
        if Global_Limit_Count > GL_limit:
            Global_Limit_Count = 0
            new_position = global_leader_decision_phase(solution, MG, k)
        if (len(new_position) > 0):
            solution = new_position.copy()  # update solution
        k += 1
    Best_fit = overall_fit.index(np.min(overall_fit))
    BEST_SOLUTION = overall_best[Best_fit]

    return int(abs(np.mean(BEST_SOLUTION)))

