import numpy as np
from Main import FCM
from Main import awgn, Rician, Rayleigh
from Main import Dinkelbach
from Beefly_pattern_based_resource_allocation import Beefly
from Main import sms


def add_noise(channel, sinr, c_type):
    if c_type == "AWGN":
        channel = awgn.awgn(channel, sinr)

    elif c_type == "Rician":
        channel = Rician.rician_channel(channel, sinr, 5)

    else: channel = Rayleigh.rayleigh_channel(channel, sinr)

    return channel


def dBm_to_W(a):
    b = 0.001 * np.power(10, (a / 10))
    return b


def W_to_dB(a):
    b = 10 * np.log10(a)
    return b


def dB_to_W(a):
    b = np.power(10, (a / 10))
    return b


def W_to_dBm(a):
    b = 10 * np.log10(a * 1000)
    return b


def GeneratecellUEPosition(SimulationRegion, N_CU):
    global CU_Position_x
    global CU_Position_y
    CU_Position_x = np.random.uniform((-SimulationRegion / 2), (SimulationRegion / 2), N_CU)
    CU_Position_y = np.random.uniform((-SimulationRegion / 2), (SimulationRegion / 2), N_CU)


def GenerateD2DPosition(SimulationRegion, N_D2D):
    global D2D_Position_x
    global D2D_Position_y
    D2D_Position_x = np.random.uniform((-SimulationRegion / 2), (SimulationRegion / 2), N_D2D)
    D2D_Position_y = np.random.uniform((-SimulationRegion / 2), (SimulationRegion / 2), N_D2D)


def Distance(x, y):
    distpow = np.power(x, 2) + np.power(y, 2)
    Dist = np.sqrt(distpow)
    return Dist


def transmission(channel_gain, power):
    # Transmission with fixed power and channel gain
    received_power = power * channel_gain
    return received_power


def Pathloss(d, PLfactor):
    Loss = np.power(d, -PLfactor)
    return Loss


def cell_D2D_dis(x1, y1, x2, y2):
    x1m = np.tile(x1, (len(x2), 1))
    y1m = np.tile(y1, (len(x2), 1))
    x2m = np.tile(x2, (len(x1), 1)).transpose()
    y2m = np.tile(y2, (len(x1), 1)).transpose()
    dis = np.sqrt((x1m - x2m) ** 2 + (y1m - y2m) ** 2)
    return dis


class Hetnet:
    def __init__(self, N_D2D, N_CU, D2D_dis, SimulationRegion, W, PLfactor, PL_k,
                       CU_tr_Power, CU_min_SINR, D2D_tr_Power_levels, D2D_tr_Power_max, D2D_min_SINR,
                       channel_type, n_cluster):
        self.N_D2D = N_D2D
        self.N_CU = N_CU
        self.D2D_dis = D2D_dis
        self.SimulationRegion = SimulationRegion
        self.W = W
        self.PLfactor = PLfactor
        self.PL_k = PL_k
        self.CU_tr_Power = CU_tr_Power
        self.CU_min_SINR = CU_min_SINR
        self.D2D_tr_Power_levels = D2D_tr_Power_levels
        self.D2D_tr_Power_max = D2D_tr_Power_max
        self.D2D_min_SINR = D2D_min_SINR
        self.collision_counter = 0
        self.collision_indicator = 0
        self.accessed_CUs = np.zeros(self.N_CU)
        self.power_levels = []
        self.CU_index = []
        self.channel_type = channel_type
        self.cluster = n_cluster                 # No of users group

        self.action_space = np.array(range(0, self.D2D_tr_Power_levels * self.N_CU))  # 10 power levels * number of CU
        self.n_actions = len(self.action_space)

        GeneratecellUEPosition(self.SimulationRegion, self.N_CU)
        GenerateD2DPosition(self.SimulationRegion, self.N_D2D)

        # ------ User Grouping / User mode Selection ------
        self.d_U, self.r_U, self.d_ind, \
            self.r_ind = FCM.cluster(self.N_CU, self.cluster)

    def reset(self):

        def cal_metrics(channel_capacity, band, u_f, Power):

            T_power = np.sum(np.array(Power))
            throughput = np.mean(channel_capacity * band * u_f)

            energy_efficiency = np.mean(throughput / T_power)
            return energy_efficiency, throughput

        # initialize power levels and construct action space
        self.power_levels = np.arange(1, self.D2D_tr_Power_levels + 1)

        # ------------ In Dedicated mode user power allocation ------------
        self.power_levels1 = []

        d_X, d_Y = [], []
        for i in range(0, len(self.d_U)):
            self.power_levels1.append(self.power_levels[self.d_ind[i]])
            self.power_levels1[i] = Dinkelbach.dinkelbach_power_allocation(self.D2D_tr_Power_levels,
                                                                           self.D2D_tr_Power_max, self.power_levels1[i])
            d_X.append(CU_Position_x[self.d_ind[i]])
            d_Y.append(CU_Position_y[self.d_ind[i]])

        d_iB = Distance(d_X, d_Y)
        CellUE_PL = Pathloss(d_iB, self.PLfactor)
        CellUE_ffading = np.random.exponential(1, size=len(self.d_U))
        CellUE_sfading = np.random.lognormal(0, dB_to_W(8), size=len(self.d_U))
        g = add_noise(self.PL_k * CellUE_PL * CellUE_ffading * CellUE_sfading, self.CU_min_SINR, self.channel_type)

        # Simulate transmission and reception
        transmission(g, self.power_levels1)

        # ------------ In Reuse mode user power allocation------------
        r_X, r_Y = [], []
        for i in range(0, len(self.r_U)):
            r_X.append(CU_Position_x[self.r_ind[i]])
            r_Y.append(CU_Position_y[self.r_ind[i]])

        self.power_levels2 = sms.resource_allocate(self.r_U, r_X, r_Y, self.D2D_tr_Power_levels)

        self.throughput = np.arange(0, self.D2D_tr_Power_levels + 1)
        for i in range(0, len(self.r_U)):
            self.power_levels2[i] = Beefly.beefly(self.D2D_tr_Power_max, self.N_D2D,
                                                    self.power_levels2[i], self.throughput[i])

        self.CU_index = np.arange(self.N_CU)
        self.action_space = self.action_space = np.transpose(
            [np.tile(self.power_levels, len(self.CU_index)), np.repeat(self.CU_index, len(self.power_levels))])

        # -----------------    channel gain within D2D j  -----------------
        D2D_Dis = self.D2D_dis
        D2D_PL = D2D_Dis ** -self.PLfactor
        D2D_ffading = np.random.exponential(1, size=self.N_D2D)
        D2D_sfading = np.random.lognormal(0, dB_to_W(8), size=self.N_D2D)
        g_j = np.tile(self.PL_k * D2D_PL, (self.N_D2D)) * D2D_ffading * D2D_sfading
        g_j = add_noise(g_j, self.D2D_min_SINR, self.channel_type)

        # -----------------    channel gain between D2D j and CU i -----------------
        d_ij = cell_D2D_dis(CU_Position_x, CU_Position_y, D2D_Position_x, D2D_Position_y)
        G_ij_ffading = np.random.exponential(1, size=self.N_CU * self.N_D2D).reshape(self.N_D2D, self.N_CU)
        G_ij_sfading = np.random.lognormal(0, dB_to_W(8), size=self.N_CU * self.N_D2D).reshape(self.N_D2D, self.N_CU)
        G_ij = self.PL_k * d_ij ** -self.PLfactor * G_ij_ffading * G_ij_sfading * 10 ** -2  # it's a matrix
        G_ij = add_noise(G_ij, self.D2D_min_SINR, self.channel_type)

        # -----------------    channel gain between D2D j and BS  -----------------
        d_jB = Distance(D2D_Position_x, D2D_Position_y)
        g_jB_ffading = np.random.exponential(1, size=self.N_D2D)
        g_jB_sfading = np.random.lognormal(0, dB_to_W(8), size=self.N_D2D)
        g_jB = self.PL_k * d_jB ** -self.PLfactor * g_jB_ffading * g_jB_sfading * 10 ** -3
        g_jB = add_noise(g_jB, self.D2D_min_SINR, self.channel_type)

        # -----------------    channel gain between D2D j and D2D j'  -----------------
        g_jj_ffading = np.random.exponential(1, size=self.N_D2D)
        g_jj_sfading = np.random.lognormal(0, dB_to_W(8), size=self.N_D2D)
        d_J_j = np.zeros(self.N_D2D)
        g_J_j = np.zeros(self.N_D2D)
        G_j_j = np.zeros(shape=(self.N_D2D, self.N_D2D))
        for j in range(self.N_D2D):
            for j_ in range(self.N_D2D):
                d_J_j[j_] = np.sqrt((D2D_Position_x[j_] - D2D_Position_x[j]) ** 2 + (
                            D2D_Position_y[j_] - D2D_Position_y[
                        j]) ** 2)  # size(d_jj) = (self.N_D2D) * 1   actually self.N_D2D-1
                if d_J_j[j_] == 0:
                    g_J_j[j_] = 0
                else:
                    g_J_j[j_] = self.PL_k * d_J_j[j_] ** -self.PLfactor * g_jj_ffading[j_] * g_jj_sfading[j_]
            G_j_j[:, j] = g_J_j  # matrix

        e, th = cal_metrics(G_ij, self.W, self.PL_k, self.power_levels2)

        return e, th

    def CU_SINR_no_collision(self, g_iB, All_D2D_Power, g_jB, All_D2D_CU_index):
        self.accessed_CUs = np.zeros(self.N_CU)
        SINR_CU = np.zeros(self.N_CU)
        for i in range(self.N_CU):
            flag_0 = 0
            flag_1 = 0
            for j in range(self.N_D2D):
                if flag_0 == 1:
                    break
                for j_ in range(self.N_D2D):
                    if j != j_ and i == All_D2D_CU_index[j] == All_D2D_CU_index[j_]:  # if multiple D2Ds choose one and the same CU i

                        SINR_CU[i] = (dBm_to_W(self.CU_tr_Power) * g_iB[i]) / (dBm_to_W(0.5))
                        flag_0 = 1
                        self.accessed_CUs[i] = 2
                        break
                else:
                    if i == All_D2D_CU_index[j]:  # define which D2D j choose CU i
                        SINR_CU[i] = (dBm_to_W(self.CU_tr_Power) * g_iB[i]) / (
                                    dBm_to_W(0.5) + All_D2D_Power[j] * g_jB[j])
                        flag_1 = 1
                        self.accessed_CUs[i] = 1
                        break
                    else:
                        continue
            if flag_1 == 0 and flag_0 == 0:
                SINR_CU[i] = (dBm_to_W(self.CU_tr_Power) * g_iB[i]) / (dBm_to_W(0.5))  # if no D2D chooses i
                self.accessed_CUs[i] = 0
        return SINR_CU

    # one D2D user can only choose one CU, i.e., if D2D users choose the same CU, reward or SINR = 0.
    def D2D_SINR_no_collision(self, All_D2D_Power, g_j, G_ij, G_j_j, All_D2D_CU_index, s):
        self.g_iJ = np.zeros(self.N_D2D)
        for j in range(self.N_D2D):
            self.g_iJ[j] = G_ij[j, All_D2D_CU_index[j]]  # for all D2D j choose CU i
        SINR_D2D = np.zeros(self.N_D2D)
        for j in range(self.N_D2D):
            for j_ in range(self.N_D2D):
                if j != j_ and All_D2D_CU_index[j] == All_D2D_CU_index[j_]:  # if collision occurs, SINR_D2D = 0
                    SINR_D2D[j] = 0
                    break
            else:
                SINR_D2D[j] = (All_D2D_Power[j] * g_j[j]) / (
                            dBm_to_W(0.5) + dBm_to_W(self.CU_tr_Power) * self.g_iJ[j])
        return SINR_D2D

    def state(self, SINR_CU):
        s = np.zeros(self.N_CU, dtype=np.float32)
        s = SINR_CU / 10 ** 10
        return s

    # one D2D user can only choose one CU, i.e., if D2D users choose the same CU, reward or SINR = 0.
    def D2D_reward_no_collision(self, SINR_D2D, SINR_CU, All_D2D_CU_index, d_ij):
        # print('all_D2D: ', All_D2D_CU_index)
        r = np.zeros(self.N_D2D)
        D2D_r = np.zeros(self.N_D2D)
        CU_r = np.sum(self.W * np.log2(1 + SINR_CU))
        for j in range(self.N_D2D):
            for j_ in range(self.N_D2D):
                if j != j_ and All_D2D_CU_index[j] == All_D2D_CU_index[ j_]:  # if multiple D2Ds choose the same CU, # r = -0.2*(10**10) collision
                    self.collision_indicator += 1
                    D2D_r[j] = -0.2 * (10 ** 10)
                    #                    CU_r = 0
                    r[j] = -0.2 * (10 ** 10)
                    break
            else:
                if SINR_CU[All_D2D_CU_index[j]] < dB_to_W(self.CU_min_SINR) or SINR_D2D[j] < dB_to_W(
                        self.D2D_min_SINR):  # if the selection of the D2D j is ith CU which is under the threshold, r = -0.1*(10**10):
                    D2D_r[j] = -0.2 * (10 ** 10)
                    r[j] = -0.2 * (10 ** 10)
                else:
                    D2D_r[j] = self.W * np.log2(1 + SINR_D2D[j])
                    r[j] = D2D_r[j] + CU_r  # - (d_ij[j][All_D2D_CU_index[j]] * (10**6))
        Net_r = sum(r)
        return r, Net_r, D2D_r, CU_r


def call_model(channel_type, users,  n_cluster, ENERGY, THROUGHPUT):
    print("\n Beefly pattern based resource allocation Running ...")

    N_D2D = users * 2                      # No of communications
    N_CU = users                           # No of cellular users
    D2D_dis = 50
    SimulationRegion = 1000                # radius 500m
    W = 10 * 10 ** 6                       # 10MHz (Band width)
    PLfactor = 4 + N_CU
    PL_k = 10 ** -2
    CU_tr_Power = 22                       # 20dBm = 100 mW = 0.1W
    CU_min_SINR = 6                        # threshold 6dB
    D2D_tr_Power_levels = 10 + N_CU
    D2D_tr_Power_max = 20
    D2D_min_SINR = 6

    ch = Hetnet(N_D2D, N_CU, D2D_dis, SimulationRegion, W, PLfactor, PL_k,
                      CU_tr_Power, CU_min_SINR, D2D_tr_Power_levels, D2D_tr_Power_max, D2D_min_SINR,
                      channel_type, n_cluster)
    E, Th = ch.reset()
    ENERGY.append(E)
    THROUGHPUT.append(Th)

    return ENERGY, THROUGHPUT


