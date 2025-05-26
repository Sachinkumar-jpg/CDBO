import numpy as np


def Algorithm(P_max, N_D2D, P_ini, throuhput):
    # Initialization
    def initial_soln(p, lb, ub, M):
        soln = []
        for i in range(p):
            t = []
            for j in range(M):
                t.append(lb + np.random.uniform() * (ub - lb))  # Equation 6
            soln.append(t)
        return soln

    def fitness(soln):
        Fit = []
        for i in range(len(soln)):
            F = 0
            for j in range(len(soln[i])):
                hr = np.random.random()
                F += soln[i][j] * hr
            Fit.append(F)
        return Fit

    def fitness1(N, P, throuhput):
        T = throuhput

        Fitness = [1 / 2 * ((1 - T) + P)]
        Fitness = np.resize(Fitness, (N))

        return Fitness

    # Update K based on Equation 9
    def update(N, D, Soln, gbest):
        k = np.random.randint(1, N)

        new_soln = []
        for i in range(N-1):
            Fi = np.mean(Soln[i])
            Fk = np.mean(Soln[k])

            # Calculating new status of population members in search space
            if (Fi > Fk):
                I = np.round(1+np.random.uniform(1, 2))                         # Equation 11

                tem = []
                for j in range(D):
                    r = np.random.randint(1, 2)
                    tem.append(Soln[i][j] + r * (gbest - I))
                new_soln.append(tem)

            else:
                tem = []
                for j in range(D):
                    i = i+1
                    r = np.random.randint(0, 2)
                    tem.append(Soln[i][j] + r * (Soln[i][j] - np.mean(gbest)))
                    i = i-1
                new_soln.append(tem)

        for i in range(len(new_soln)):
            a = np.array(new_soln[i])
            shape = len(a.shape)
            if shape == 2:
                for j in range(len(a)):
                    a[j] = np.mean(a[j])
                    new_soln[i] = a[j]

        return new_soln

    g = 1
    N = N_D2D                  # population size
    T = 200                    # Max iteration
    lb, ub = 1, P_max          # Lower bound and upper bound
    M = 10                     # number of decision variable
    Solution = initial_soln(N, lb, ub, M)

    Fitness = fitness1(N_D2D, P_ini, throuhput)             # Fitness of solution
    best_fit = np.min(Fitness)                              # Best Fitness
    best = np.argmin(Fitness)                               # Index of Best Fit
    best_soln = Solution[best]                              # Best solution
    overall_fit, overall_best = [], []

    while g < T:
        new_soln = update(N, M, Solution, best_soln)
        Fitt = fitness(new_soln)            # Fitness of solution
        new_fit = np.min(Fitt)              # Best Fitness
        best = np.argmin(Fitt)              # Index of Best Fit
        best_soln = Solution[best]          # Best solution

        if new_fit < best_fit:
            best_fit = new_fit.copy()
            best = np.argmin(Fitt)          # Index of Best Fit
            best_soln = Solution[best]      # Best solution
        else:
            new_soln = update(N, M, Solution, best_soln)
            Fitt = fitness(new_soln)        # Fitness of solution
            new_fit = np.min(Fitt)          # Best Fitness
            best = np.argmin(new_fit)       # Index of Best Fit
            best_soln = Solution[best]      # Best solution

        overall_fit.append(best_fit)
        overall_best.append(best_soln)

        g += 1
    best = np.argmin(overall_fit)
    BEST_SOLUTION = overall_best[best]

    return np.mean(BEST_SOLUTION)
