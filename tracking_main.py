from matplotlib import pyplot as plt
import numpy as np


def kalman_predict(x, P, F, Q):
    xp = F * x
    Pp = F * P * F.T + Q
    return xp, Pp


def kalman_update(x, P, H, R, z):
    S = H * P * H.T + R
    K = P * H.T * np.linalg.inv(S)
    zp = H * x

    xe = x + K * (z - zp)
    Pe = P - K * H * P
    return xe, Pe


def kalman_tracking(
    z,
    x01=0.0,
    x02=0.0,
    x03=0.0,
    x04=0.0,
    dt=0.5,
    nx=0.16,
    ny=0.36,
    nvx=0.16,
    nvy=0.36,
    nu=0.25,
    nv=0.25,
):
    # Constant Velocity
    F = np.matrix([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    # Cartesian observation model
    H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])

    # Motion Noise Model
    Q = np.matrix([[nx, 0, 0, 0], [0, nvx, 0, 0], [0, 0, ny, 0], [0, 0, 0, nvy]])

    # Measurement Noise Model
    R = np.matrix([[nu, 0], [0, nv]])

    x = np.matrix([x01, x02, x03, x04]).T
    P = Q

    N = len(z[0])
    s = np.zeros((4, N))
    # s[0][0] = x01
    # s[2][0] = x02

    for i in range(N):
        xp, Pp = kalman_predict(x, P, F, Q)
        x, P = kalman_update(xp, Pp, H, R, z[:, i])
        val = np.array(x[:2, :2]).flatten()
        s[:, i] = val

    px = s[0, :]
    py = s[1, :]

    return px, py


def rms(x, y, px, py):
    err = np.array(error(x, y, px, py))
    print("mean =", err.mean())
    print("std =", err.std())
    print("rms =", np.sqrt(err.mean()))
    return np.sqrt(err.mean())


def error(x, y, px, py):
    err = []
    for i in range(len(x)):
        err.append(np.sqrt((x[i] - px[i]) ** 2 + (y[i] - py[i]) ** 2))
    return err


# def optimisation(trial, x, y, z, dt, nx, ny, nvx, nvy, nu, nv, x01, x02, x03, x04):
#     # dt = trial.suggest_float("dt", 0.05, 1.0, step=0.05)
#     # Q
#     nx = trial.suggest_float("nx", 0.0, 200.0)
#     ny = trial.suggest_float("ny", 0, 0.0001)
#     nvx = trial.suggest_float("nvx", 0, 0.0001)
#     nvy = trial.suggest_float("nvy", 0, 0.0001)

#     # R
#     nu = trial.suggest_float("nu", 0, 0.0001)
#     nv = trial.suggest_float("nv", 0, 0.0001)

#     # init x
#     x01 = z[0][0]
#     x02 = z[1][0]

#     px, py = kalman_tracking(z, x01, x02, x03, x04, dt, nx, ny, nvx, nvy, nu, nv)
#     rms_val = rms(x, y, px, py)
#     return rms_val


# def optimize_rms(x, y, z):
#     import optuna
#     from tqdm import tqdm

#     trials = 20000

#     pbar = tqdm(total=trials, desc="Optimization Progress")

#     def print_new_optimal(study, trial):
#         # Check if the trial is better than the current best
#         pbar.update(1)
#         if trial.value == study.best_value:
#             print(f"New Best RMS: {trial.value} (trial number {trial.number})")
#             print("Best parameters:", study.best_params)

#     optuna.logging.set_verbosity(optuna.logging.WARNING)

#     study = optuna.create_study()
#     dt = 0.5
#     # nx = 0.16
#     # ny = 0.36
#     # nvx = 0.16
#     # nvy = 0.36
#     # nu = 0.25
#     # nv = 0.25
#     x01 = z[0][0]
#     x02 = z[1][0]

#     # nx = -90.0
#     # ny = 203.0
#     # nvx = -13.0
#     # nvy = 720.0
#     # nu = -23.0
#     # nv = -5.0
#     # x01 = 190.7
#     # x02 = 43.2
#     # x03 = -842.5
#     # x04 = 38.58

#     # nx = 265.6984895041111
#     # ny = 4377.212482966368
#     # nvx = 0.1650447124433483
#     # nvy = 694.8165035716708
#     # nu = 23.167412752765156
#     # nv = -13.172081648387536
#     # x01 = -372.1342841527826
#     # x02 = 441.2537740166025
#     # x03 = 395.294325684342
#     # x04 = -37.807362565797575
#     #
#     # nx = 1979.7581258369369
#     # ny = 3367.2277769762745
#     # nvx = 1.3889333137895448
#     # nvy = 4869.619368772957
#     # nu = -1.807557642508617
#     # nv = -11.992831691039886
#     # x01 = 590.1013308245263
#     # x02 = 447.4390004186095
#     # x03 = 254.65467776450595
#     # x04 = 973.5004268095914
#     #
#     # nx = 986.2429185451157
#     # ny = 5919.531760939467
#     # nvx = 0.7169563727200654
#     # nvy = 5064.198891444827
#     # nu = 2.8079222224627323
#     # nv = -16.110942523796034
#     # x01 = 584.8417681605206
#     # x02 = 448.64207035865655
#     # x03 = 59.73328857781342
#     # x04 = 763.0144886665084

#     nx = 40.0
#     ny = 9e-05
#     nvx = 4e-05
#     nvy = 9e-05
#     nu = 6.25e-05
#     nv = 6.25e-05

#     study.optimize(
#         lambda trial: optimisation(
#             trial, x, y, z, dt, nx, ny, nvx, nvy, nu, nv, x01, x02, x03, x04
#         ),
#         n_trials=trials,
#         n_jobs=8,
#         callbacks=[print_new_optimal],  # Add the callback here
#     )

#     return study.best_params


if __name__ == "__main__":

    x = np.genfromtxt("data/x.csv", delimiter=",")
    y = np.genfromtxt("data/y.csv", delimiter=",")
    na = np.genfromtxt("data/na.csv", delimiter=",")
    nb = np.genfromtxt("data/nb.csv", delimiter=",")
    z = np.stack((na, nb))

    dt = 0.5
    # tune 1
    # nx = 0.16
    # ny = 0.36
    # nvx = 0.16
    # nvy = 0.36
    # nu = 0.25
    # nv = 0.25
    # scales = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # tune 2
    nx = 160.0
    ny = 0.00036
    nvx = 0.00016
    nvy = 0.00036
    nu = 0.00025
    nv = 0.00025
    x03 = 0.0
    x04 = 0.0

    # scales = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    # magnitudes = [0.1, 1, 10, 100]
    # scales = [scale * magnitude for scale in scales for magnitude in magnitudes]

    k = 1
    nx = k * 0.16
    ny = k * 0.36
    nvx = k * 0.16
    nvy = k * 0.36
    nu = 0.25
    nv = 0.25
    x01 = z[0][0]
    x02 = 425

    px, py = kalman_tracking(
        nx=nx,
        ny=ny,
        nvx=nvx,
        nvy=nvy,
        nu=nu,
        nv=nv,
        x01=x01,
        # x02=x02,
        x03=x03,
        # x04=x04,
        z=z,
    )
    # error = rms(x, y, px, pypx, py)
    print("predicction rms:", rms(x, y, px, py))

    print("noise rms:", rms(x, y, na, nb))

    # prev_rms = float('inf')
    # error_data = []
    # from tqdm import tqdm
    # for kxn in tqdm(scales):
    #     for kyn in scales:
    #         for knvx in scales:
    #             for knvy in scales:
    #                 for knu in scales:
    #                     for knv in scales:
    #                         px, py = kalman_tracking(
    #                             nx=nx * kxn,
    #                             ny=ny * kxn,
    #                             nvx=nvx,
    #                             nvy=nvy,
    #                             nu=nu,
    #                             nv=nv,
    #                             x01=x01,
    #                             x02=x02,
    #                             x03=x03,
    #                             x04=x04,
    #                             z=z,
    #                         )
    #                         current_rms = rms(x, y, px, py)
    #                         error_info = {
    #                             'params': {
    #                                 'nx': nx * kxn,
    #                                 'ny': ny * kyn,
    #                                 'nvx': nvx * knvx,
    #                                 'nvy': nvy * knvy,
    #                                 'nu': nu * knu,
    #                                 'nv': nv * knv,
    #                             },
    #                             'rms': current_rms,
    #                             }
    #                         if error_info['rms'] < lowsest_rms:
    #                             lowsest_rms = error_info['rms']
    #                             print(error_info)
    #                             error_data.append(error_info)
    #                         if current_rms > prev_rms:
    #                             break
    #                         prev_rms = current_rms

    # print(error_data)

    # optimize_rms(x, y, z)

    # px, py = kalman_tracking(
    #     nx=nx,
    #     ny=ny,
    #     nvx=nvx,
    #     nvy=nvy,
    #     nu=nu,
    #     nv=nv,
    #     x01=x01,
    #     x02=x02,
    #     x03=x03,
    #     x04=x04,
    #     z=z,
    # )
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)
    plt.plot(px, py)
    plt.scatter(na, nb)
    plt.title("Kalman Filter")
    plt.savefig("Report/assets/tracking/kalman_filter.png")
    plt.show()

    """
New Best RMS: 3.036106804621608 (trial number 783)
Best parameters: {'nx': 1642.252858315786, 'ny': 1141.8251255187547, 'nvx': 1.3207315123262986, 'nvy': 6161.1442334153735, 'nu': 11.625781506098406, 'nv': -6.75098576214653, 'x01': 742.1163757977941, 'x02': 449.95163847270913, 'x03': 719.8521842924018, 'x04': 217.8430376223692}
    """

# [
# {
#     'params': {
#         'nx': 0.00016, 'ny': 0.00035999999999999997, 'nvx': 0.00016, 'nvy': 0.00035999999999999997, 'nu': 0.00025, 'nv': 0.00025},
#     'rms': 4.47764590930272
# }, {
#     'params': {
#         'nx': 0.0016, 'ny': 0.00035999999999999997, 'nvx': 0.00016, 'nvy': 0.00035999999999999997, 'nu': 0.00025, 'nv': 0.00025
#         },
#     'rms': 4.463090946055551
# }, {
#     'params': {
#         'nx': 0.016, 'ny': 0.00035999999999999997, 'nvx': 0.00016, 'nvy': 0.00035999999999999997, 'nu': 0.00025, 'nv': 0.00025
#         },
#     'rms': 4.335775909250596
# }, {
#     'params': {
#         'nx': 0.16, 'ny': 0.00035999999999999997, 'nvx': 0.00016, 'nvy': 0.00035999999999999997, 'nu': 0.00025, 'nv': 0.00025
#         },
#     'rms': 3.8692383799852865
# }, {
#     'params': {
#         'nx': 1.6, 'ny': 0.00035999999999999997, 'nvx': 0.00016, 'nvy': 0.00035999999999999997, 'nu': 0.00025, 'nv': 0.00025
#         },
#     'rms': 3.2510288733293216
# },{
#     'params': {
#         'nx': 16.0, 'ny': 0.00035999999999999997, 'nvx': 0.00016, 'nvy': 0.00035999999999999997, 'nu': 0.00025, 'nv': 0.00025
#         },
#     'rms': 2.711796606244655
# },{
#     'params': {

# 'nx': 160.0,
# 'ny': 0.00035999999999999997,
# 'nvx': 0.00016,
# 'nvy': 0.00035999999999999997,
# 'nu': 0.00025,
# 'nv': 0.00025
#         },
#     'rms': 2.6473822509699887
# }
