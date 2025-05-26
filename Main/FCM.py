import random, math, numpy as np


def cluster(User, cluster_n):

    Users = []
    for i in range(1, User):
        Users.append(i)

    Users = np.array(Users)
    Users = Users.reshape(-1, 1)

    U, obj_fcn, lamda, beta = [], [], 1, 1

    def initfcm(cluster_n, data_n):
        U_part, U_transpose = [], []
        col_sum, col_sum_change = [], []
        for i in range(cluster_n):
            tem = []
            for j in range(data_n):
                tem.append(random.random())
            U_part.append(tem)

        U_transpose = np.transpose(U_part)
        for i in range(len(U_transpose)):
            sum = 0
            for j in range(len(U_transpose[i])):
                sum += U_transpose[i][j]
            col_sum.append(sum)

        for i in range(cluster_n):
            col_sum_change.append(col_sum)  # make col_sum size = U_part size = cluster size

        # elementwise division
        for i in range(len(U_part)):
            tem = []
            for j in range(len(U_part[i])):
                tem.append(U_part[i][j] / col_sum_change[i][j])
            U.append(tem)
        return U

    def distfcm(center, data):
        out, ones, matmul_ones_center, data_minus_ones = [], [], [], []

        # fill the output matrix
        for i in range(len(data)):
            tem = []
            for j in range(1):
                tem.append(1.0)
            ones.append(tem)  # ones of size data size x 1

        for k in range(len(center)):
            # matrix mul of ones and center or copy each row of center to data size(to make center size = data size)
            matmul_ones_center = []
            for i in range(len(data)):
                matmul_ones_center.append(center[k])

            # (data - matmul_ones_center) ^ 2 -- (elementwise)
            data_minus_ones = []
            for i in range(len(data)):
                tem = []
                for j in range(len(data[i])):
                    tem.append(math.pow((data[i][j] - matmul_ones_center[i][j]), 2))
                data_minus_ones.append(tem)

            # sum elements of row, then take square root
            tem = []
            for i in range(len(data_minus_ones)):
                sum = 0
                for j in range(len(data_minus_ones[i])):
                    sum += data_minus_ones[i][j]
                tem.append(math.sqrt(sum))
            out.append(tem)

        return out

    def summ(data):
        summ = 0
        for i in range(len(data)):
            summ += data[i]
        return summ

    def stepfcm(data, U, cluster_n, expo):
        tmp_sum, mf, matmul_mf_data, matmul_row_mf_sum_ones, row_mf_sum = [], [], [], [], []
        ones, center, dist, dist_mul_mf, tmp_sum_change, U_new = [], [], [], [], [], []

        for i in range(len(U)):
            tem = []
            for j in range(len(U[i])):
                tem.append(math.pow(U[i][j], expo))
            mf.append(tem)

        # mf * data
        for i in range(len(mf)):
            tem = []
            for j in range(len(data[0])):
                a = 0
                for k in range(len(data)):
                    a += (mf[i][k] * data[k][j])  # matrix mul of mf and data
                tem.append(a)
            matmul_mf_data.append(tem)

        for i in range(len(mf)):
            tem, sum = [], 0
            for j in range(len(mf[i])):
                sum += mf[i][j]  # adding attributes of the row to make single column
            tem.append(sum)
            row_mf_sum.append(tem)

        for i in range(1):
            tem = []
            for j in range(len(data[0])):
                tem.append(1.0)
            ones.append(tem)

        # row_mf_sum * ones
        for i in range(len(row_mf_sum)):
            tem = []
            for j in range(len(ones[0])):
                a = 0
                for k in range(len(ones)):
                    a += (row_mf_sum[i][k] * ones[k][j])  # matrix mul of row_mf_sum and ones
                tem.append(a)
            matmul_row_mf_sum_ones.append(tem)

        # elementwise division
        for i in range(len(matmul_mf_data)):
            tem = []
            for j in range(len(matmul_mf_data[i])):
                tem.append(matmul_mf_data[i][j] / matmul_row_mf_sum_ones[i][j])
            center.append(tem)

        dist = distfcm(center, data)

        a, b = 0, 0
        for i in range(len(dist)):
            summ_mf = summ(mf[i])
            b += beta * (1.0 - summ_mf)
            for j in range(len(dist[i])):
                a += (mf[i][j] * math.pow(dist[i][j], 2)) + (lamda * math.log10(mf[i][j])) + (
                        lamda * math.log10(math.pow(dist[i][j], 2))) + b
        obj_fcn.append(a)

        for i in range(len(dist)):
            tem = []
            for j in range(len(dist[i])):
                tem.append(lamda / (beta - math.pow(dist[i][j], 2)))
            U_new.append(tem)

        return U_new

    data, U_trans, Cluster = [], [], []
    data = Users                                        # Users details
    data_n = len(data)                                  # data
    expo, max_iter, min_impro = 2, 5, 1e-6              # exponent of U, max iteration, min improvement
    U = initfcm(cluster_n, data_n)                      # Initial fuzzy partition
    g = 0

    while (g < max_iter):
        U = stepfcm(data, U, cluster_n, expo)
        if (g > 1):
            if (np.abs(obj_fcn[g] - obj_fcn[g - 1]) < min_impro):           # termination condition
                g = max_iter
        g += 1

    U_trans = np.transpose(U)
    for i in range(len(U_trans)):
        Cluster.append(np.argmax(U_trans[i])+1)

    Dedicated_mode, Reuse_mode = [], []                     # Dedicated and Reuse user groups
    D_ind, R_ind = [], []

    for i in range(len(Users)):
        if Cluster[i] == 1:
            Dedicated_mode.append(Users[i])
            D_ind.append(i)
        else:
            Reuse_mode.append(Users[i])
            R_ind.append(i)

    return Dedicated_mode, Reuse_mode, D_ind, R_ind
