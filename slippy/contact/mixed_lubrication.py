""" This is just a python port of Abdullah's code should not be used or referenced in any way!"""


import numpy as np
from scipy.linalg.lapack import dgtsv


def update_pressure(relaxation_factor: float, new_pressure: np.ndarray, old_pressure: np.ndarray):
    """Update the pressure using a relaxation factor, find residual from the previous iteration

    Parameters
    ----------
    relaxation_factor: float
        Between 0 and 1, 0 meaning that the value for the pressures is not updated at all, 1 meaning that the pressures
        are exactly as found in this iteration. Smaller numbers result in more stable iterations, larger numbers in
        faster convergence
    new_pressure: np.ndarray
        Array of pressures found in this iteration
    old_pressure: np.ndarray
        Array of pressures from the previous iteration

    Returns
    -------
    relaxed_new_pressure: np.ndarray
        The new pressures after the relaxation factor has been applied
    old_pressure: np.ndarray
        A copy of the relaxed_new_pressure array
    relative_error: float
        The relative residual from this iteration

    Notes
    -----
    The relative error is the absolute total change in pressure divided by the total pressure
    """
    relaxed_new_pressure = (1 - relaxation_factor) * old_pressure + relaxation_factor * new_pressure
    relative_error = (np.mean(np.abs(relaxed_new_pressure - old_pressure).flatten()) /
                      np.mean(relaxed_new_pressure.flatten()))
    old_pressure = relaxed_new_pressure.copy()

    return relaxed_new_pressure, old_pressure, relative_error


def roughness(DA):
    """reads the roughness from a file then scales it to the right size, dosn't seem to change for different time steps
    rmsa is the Sa parameter of the roughness


    Parameters
    ----------
    DA
        Roughness amptitude

    Returns
    -------
    ROU
        Roughness function
    RMSA
        Sa parameter for the roughness
    """
    return np.zeros(10, 10), 5.1


def GEOM(N, DAB, X, Y, GOU, ROU2, RMSRB):
    """Gets the geometry of the ball adds it to the geometry of the roughness

    Parameters
    ----------
    N
    DAB
    X
    Y
    GOU
    ROU2
    RMSRB

    Returns
    -------

    """
    ROU2, RMSA = roughness(DAB)

    ROU2 = ROU2.transpose()

    x_mesh, y_mesh = np.meshgrid(X, Y)
    GOU = (x_mesh ** 2 + y_mesh ** 2) * 0.5 - ROU2

    return GOU


def guess_initial_pressures(shape_square_only, x_start, x_end):
    """Populates the pressure and old pressure with a gues of initial pressures based on somthing?

    Parameters
    ----------
    shape_square_only
    x_start
    x_end

    Returns
    -------

    """
    grid_spacing = (x_end - x_start) / (shape_square_only - 1)  # regular gid spacing
    y_start = -0.5 * (x_end - x_start)  # square domain
    x = np.linspace(x_start, x_end, shape_square_only)
    y = x - x_start + y_start

    x_mesh, y_mesh = np.meshgrid(x, y)
    c = 1 - x_mesh ** 2 - y_mesh ** 2
    p = np.zeros_like(x_mesh)

    p[c > 0] = np.sqrt(c[c > 0])
    p_old = p.copy()

    return grid_spacing, x, y, p, p_old


def tdma(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side):
    """The thomas algorthim (TDMA) solution for tridiagonal matrix inversion

    Parameters
    ----------
    lower_diagonal: np.ndarray
        The lower diagonal of the matrix length n-1
    main_diagonal: np.ndarray
        The main diagonal of the matrix length n
    upper_diagonal: np.ndarray
        The upper digonal of the matrix length n-1
    right_hand_side: np.ndarray
        The array bof size max(1, ldb*nrhs) for column major layout and max(1, ldb*n) for row major layout contains the
        matrix B whose columns are the right-hand sides for the systems of equations.

    Returns
    -------
    x: np.ndaray
        the solution array length n

    Notes
    -----
    Nothing is mutated by this function
    """
    _, _, _, x, _ = dgtsv(lower_diagonal, main_diagonal, upper_diagonal, right_hand_side)
    return x


def VI(DX, p, AK):
    """solves the normal contact problem

    Parameters
    ----------
    DX: float
        Grid spacing
    p: np.array
        Pressures
    AK: np.array
        The influence matrix

    Returns
    -------
    V: np.ndarray
        Array of surface displacemetns
    """
    return np.zeros_like(p)


def populate_influence_matrix(num_points):
    """get influence matrix components
    
    Parameters
    ----------
    num_points: int
        The number of points in each direction

    Returns
    -------
    AK: np.array
        The influence matrix (needs multiplying by 2/pi**2 to be used
    """
    s = lambda x, y: x + np.sqrt(x ** 2 + y ** 2)

    ak = np.zeros((num_points, num_points))

    for i in range(num_points):
        xp = i + 0.5
        xm = i - 0.5
        for j in range(i + 1):
            ym = j - 0.5
            yp = j + 0.5
            a1 = s(yp, xp) / s(ym, xp)
            a2 = s(xm, ym) / s(xp, ym)
            a3 = s(ym, xm) / s(yp, xm)
            a4 = s(xp, yp) / s(xm, yp)
            ak[i, j] = xp * np.log(a1) + ym * np.log(a2) + xm * np.log(a3) + yp * np.log(a4)
            ak[j, i] = ak[i, j]
    return ak


def ITERSEMISEPI(N: int, KK: int, DX: float, DT: float, ERH, H00, G0, X, Y, H, HO0, RO, RO0, EPS, EDA, P, V, GOU, ROU,
                 ENDA, A1, A2, A3, roelands_exponent, HM0, HM0AVG, DH, DW,
                 AK, E1, RX, B, PH, Hardness, UPLASTIC):
    # reynolds solver start, put all the reynolds solver in a single func

    AK00 = 2 / np.pi ** 2 * AK[0, 0] # replace these with their definition,
    AK10 = 2 / np.pi ** 2 * AK[1, 0]
    AK20 = 2 / np.pi ** 2 * AK[2, 0]  # first three components of the influence matrix

    DXT = 1 / DT if DT > 0 else 0.0

    DX1 = 1 / DX
    DX2 = DX ** 2
    DX3 = 1 / DX2
    DX3RHO = DX3 / RO

    ID = 0

    AP, BP, CP, FP = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    AW, BW, CW, FW = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    ATIME, BTIME, CTIME, FTIME = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    for K in range(KK):
        ICB, ICM, ICP, ICT = 0, 0, 0, 0
        for J in range(N - 1, 1, -1):

            for I in range(1, N):
                D1 = 0.5 * (EPS[I - 1, J] + EPS[I, J])  # CHANGED from original...
                D2 = 0.5 * (EPS[I + 1, J] + EPS[I, J])
                D4 = 0.5 * (EPS[I, J - 1] + EPS[I, J])
                D5 = 0.5 * (EPS[I, J + 1] + EPS[I, J])
                D3 = D1 + D2 + D4 + D5

                Q1 = AK10 * P[I - 1, J] + AK00 * P[I, J] + AK10 * P[I + 1, J]
                Q2 = AK00 * P[I - 1, J] + AK10 * P[I, J] + AK20 * P[I + 1, J]

                AP[I] = D1 * DX3RHO(I, J)
                BP[I] = -1 * D3 * DX3RHO(I, J)
                CP[I] = D2 * DX3RHO(I, J)
                FP[I] = -1 * (D5 * P[I, J + 1]) + D4 * P[I, J - 1] * DX3RHO(I, J)

                AW[I] = (AK00 - AK10) * DX1
                BW[I] = (AK10 - AK00) * DX1
                CW[I] = (AK20 - AK10) * DX1
                FW[I] = (((H[I, J] - Q1) - (H[I - 1, J] - Q2)) * DX1 +
                         H[I, J] * (1 - (RO[I - 1, J] / RO[I, J])) * DX +
                         DX1 * (ROU[I, J] - ROU[I - 1, J]))

                ATIME[I] = -1 * AK10 * DXT
                BTIME[I] = -1 * AK00 * DXT
                CTIME[I] = -1 * AK10 * DXT
                FTIME[I] = ((H[I, J] - Q1) - (RO0[I, J] / RO[I, J]) * HO0[I, J]) * DXT

                if H[I, J] <= 0.47e-9 * RX / B / B:
                    ICM += 1

            A = AP + AW + ATIME
            B = BP + BW + BTIME
            C = CP + CW + CTIME
            F = FP + FW + FTIME

            B[0] = 1.0
            B[N] = 1.0
            A[N] = 0.0
            C[0] = 0.0
            F[0] = 0.0
            F[N] = 0.0

            P1D = tdma(A, B, C, F)

            P1D[P1D < 0] = 0.0

            P[:, J] = P1D

        # reynolds solver end


        # HREEI(** continue from here!!!!!!!!!!)
    # write results return outputs
    return


def HREEI(DX, p, AK, GOU, ROU, KK, G0, A1, A2, A3, roelands_exponent, ENDA):
    """iterate through other sub systems, get rid of this function instead call sub models from the fluid

    KK: some flag?
    p: pressures from last iteration
    gou:?
    ak: influence matrix components
    rou: roughness
    dx: grid spacing
    G0: some dimentionality thing for the pressure result?

    """

    PAI = np.pi
    PAI1 = 2 / np.pi ** 2

    V = VI(DX, p, AK)  # deflection of surfaces, should this be in here?

    some_flag = False

    # find HMIN
    H = (GOU + ROU + V)  # gap height
    HMIN = min(H).flatten()

    if KK == 0:
        KK = 1
        DH = 0.005 * HMIN  # change in height per iteration?
        H00 = 0.0  # output is written here

    # adjust for load balance
    W1 = np.sum(p)
    W1 *= DX ** 2 / G0 # < total load (non dimentional)
    DW = 1.0 - W1
    H00OLD = H00.copy()
    H00 -= 0.1 * DW
    ERH = H00 - H00OLD
    H += H00

    # pressure viscosity
    EDA = np.exp(A1 * (-1.0 + (1.0 + A2 * p) ** roelands_exponent))

    # density
    RO = (A3 + 1.34 * p) / (A3 + p)
    EPS = RO * H ** 3 / (ENDA * EDA)

    return H, EDA, RO, EPS


def ehl(n, n1, PH, E1, eta_0, ball_radius, US, X0, XE, sliding_to_rolling_ratio, grid_spacing):
    """

    Parameters
    ----------
    original inputs:
    n
    n1

    read from input file:
    PH
    E1
    eta_0: was EDA0
    ball_radius: was Rx
    US
    X0
    XE
    sliding_to_rolling_ratio

    Returns
    -------

    """
    pi = np.pi
    roelands_exponent = 0.68
    P0 = 1.96e8

    PLCOUNT = np.zeros((n, n), dtype=int)

    Hsub = 1.15e9 / PH
    Hardness = Hsub

    MM = N - 1

    A1 = np.log(eta_0) + 9.67
    A2 = 5.1e-9 * PH
    A3 = 0.59 / (PH * 1.0e-9)

    # Setting the speeds
    ball_speed = 0.253  # speed of the ball
    flat_speed = 0.248  # speed of the disc
    rolling_speed = (ball_speed + flat_speed) / 2
    sliding_to_rolling_ratio = 2 * (ball_speed - flat_speed) / (ball_speed + flat_speed)  # sliding to rolling ratio
    nd_rolling_speed = eta_0 * rolling_speed / (E1 * ball_radius)  # non dimentional speed

    hertzian_half_width = pi * PH * ball_radius / E1  # Half width of hertzian contact
    nd_load = 2 * pi * PH / (3 * E1) * (hertzian_half_width / ball_radius)  # nondimentioal load
    W = nd_load * E1 * ball_radius ** 2  # load in newtons

    ALFA = 14.94e-9  # Pressure viscosity
    G = ALFA * E1

    HM0 = 3.63 * (ball_radius / hertzian_half_width) ** 2 * G ** 0.49 * nd_rolling_speed ** 0.68 * nd_load ** (-0.073) * (
                1 - np.exp(-0.68))

    lambda_bar = 12 * rolling_speed * eta_0 * ball_radius ** 2 / (
                hertzian_half_width ** 3 * PH)  # 12*nd_rolling_speed*(E1/PH)*(ball_radius/hertzian_half_width)**3  # reynolds ..somthing?
    # was lambda_bar

    # write some stuff to file

    G0 = 2.0943951

    initi(N, grid_spacing, X0, XE, X, Y, P, POLD)

    DT = 0.0  # delta t
    DAD = 0.0  # A disc
    DAB = 0.0  # A ball

    roughness(N, DAD, ROU, RMSR)  # finds the roughness function
    geom(N, DAB, X, Y, GOU, ROU2, RMSRB)  # adds the roughness to the profile

    # this is the main loop here

    relaxation_factor = 0.2  # relaxation factor
    KK = 0

    HREEI(N, grid_spacing, DF, DT, ERH, KK, H00, G0, X, Y, H, RO, EPS, EDA, P, V, GOU, ROU)

    # this seems to be the loop that actually solves things

    MK = 1
    while True:
        ER = itersemisepi(n, KK, grid_spacing, DT, ERH, H00, G0, X, Y, H, HO0, RO, RO0, EPS, EDA, P, V, GOU, ROU)
        MK += 1
        update_pressure(N, relaxation_factor, ER, P, POLD)
        if ER < 1e-6:
            break
        if MK == 15:
            MK = 1
            DH *= 0.75
            if DW < 1e-4:
                DH *= 0.9

    # END of the main loop

    # calculating maximum pressure and minimum film thickness

    P2 = np.max(P.flatten())  # ND maximum pressure (no roughness?)
    min_possible_height = 1e-9 * ball_radius / hertzian_half_width / hertzian_half_width
    H[H < min_possible_height] = min_possible_height
    H2 = min(H.flatten())  # ND minimum film thickness (no roughness?)
    H3 = H2 * hertzian_half_width * hertzian_half_width / ball_radius  # minimum film thickness in m
    P3 = P2 * PH  # Max pressure in pascals

    # Finding the load sharing

    LASPT = np.sum(P[
                       H <= 1e-9 * ball_radius / hertzian_half_width / hertzian_half_width])  # ND sum of pressures taken by solid to solid contact
    LASPT = LASPT * grid_spacing * grid_spacing / G0  # dimentionalised load taken by solid to solid contact

    ####################################################################################################################
    # End of calculation without roughness
    ####################################################################################################################

    # to move roughness change DT to be greater than 0
    # DAD disc roughness amptitude
    # DAB ball roughness amptitude
    # DF does somthing with single asperity contacts, not active in this simulation

    DT = 0.0
    DAD = 0.0
    DAB = 0.0
    G00 = 0.0
    G001 = 0.0

    roughness(N, DAD, ROU, RMSR)
    GEOM(N, DAB, X, Y, GOU, ROU2, RMSRB)

    for K2 in range(5):
        A2 = 5.1e-9 * PH
        A3 = 0.59 / (PH * 1e-9)

        ball_speed = 0.2530  # ball speed
        flat_speed = 0.2480

        sliding_to_rolling_ratio = 2 * (ball_speed - flat_speed) / (ball_speed + flat_speed)
        nd_rolling_speed = eta_0 * rolling_speed / (E1 * ball_radius)

        nd_load = 2 * pi * PH / (3 * E1) * (hertzian_half_width / ball_radius) ** 2
        W = nd_load * E1 * ball_radius ** 2

        HM0 = 3.63 * (ball_radius / hertzian_half_width) ** 2 * G ** 0.49 * nd_rolling_speed ** 0.68 * nd_load ** (
            -0.073) * (1 - np.exp(-0.68))
        HMC = 2.69 * (ball_radius / hertzian_half_width) ** 2 * G ** 0.53 * nd_rolling_speed ** 0.67 * nd_load ** (
            -0.067) * (1 - 0.61 * np.exp(-0.73))

        lambda_bar = 12 * rolling_speed * eta_0 * ball_radius ** 2 / (hertzian_half_width ** 3 * PH)

        MK = 1
        G0 = 2.0943951

        FZ(N, H, HO0)
        FZ(N, RO, RO0)

        # Start of the main loop again

        HREEI(N, grid_spacing, DF, DT, ERH, KK, H00, G0, X, Y, H, RO, EPS, EDA, P, V, GOU, ROU)

        relaxation_factor = 0.2

        COUNT = 0

        while True:
            COUNT += 1
            KK = 20
            H00initial = H00
            ER = itersemisepi(n, KK, grid_spacing, DT, ERH, H00, G0, X, Y, H, HO0, RO, RO0, EPS, EDA, P, V, GOU, ROU)
            MK += 1
            update_pressure(n, relaxation_factor, ER, P, POLD)
            # some output here
            if ER > 1e-6:
                if MK >= 15:
                    MK = 0
                    DH *= 0.9
                    if DW < 1e-5:
                        DH *= 0.95
            else:
                if COUNT >= 70:
                    break

        # Main loop ends here

        # End of main loop

        # FInd max pressure and min film thickness

        # Find load sharing

        # find average, deformed gap


if __name__ == '__main__':
    N = int(65)
    N1 = int(1)

    subak(N - 1)
    ehl(N, N1)
