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
    kq=1,
    kr=1,
):
    # Constant Velocity
    F = np.matrix([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    # Cartesian observation model
    H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])

    # Motion Noise Model
    Q = kq * np.matrix([[nx, 0, 0, 0], [0, nvx, 0, 0], [0, 0, ny, 0], [0, 0, 0, nvy]])

    # Measurement Noise Model
    R = kr * np.matrix([[nu, 0], [0, nv]])

    x = np.matrix([x01, x02, x03, x04]).T
    P = Q

    N = len(z[0])
    s = np.zeros((4, N))

    for i in range(N):
        xp, Pp = kalman_predict(x, P, F, Q)
        x, P = kalman_update(xp, Pp, H, R, z[:, i])
        val = np.array(x[:2, :2]).flatten()
        s[:, i] = val

    px = s[0, :]
    py = s[1, :]

    return px, py


def rms(x, y, px, py):
    return np.sqrt(1 / len(px) * (np.sum((x - px) ** 2 + (y - py) ** 2)))


def mean(x, y, px, py):
    return np.mean(np.sqrt((x - px) ** 2 + (y - py) ** 2))


if __name__ == "__main__":

    x = np.genfromtxt("data/x.csv", delimiter=",")
    y = np.genfromtxt("data/y.csv", delimiter=",")
    na = np.genfromtxt("data/na.csv", delimiter=",")
    nb = np.genfromtxt("data/nb.csv", delimiter=",")
    z = np.stack((na, nb))

    dt = 0.5
    nx = 160.0
    ny = 0.00036
    nvx = 0.00016
    nvy = 0.00036
    nu = 0.00025
    nv = 0.00025

    px1, py1 = kalman_tracking(
        z=z,
    )

    nx = 0.16 * 10
    ny = 0.36
    nvx = 0.16 * 0.0175
    nvy = 0.36 * 0.0175
    nu = 0.25
    nv = 0.25 * 0.001
    kq = 0.0175
    kr = 0.0015

    px2, py2 = kalman_tracking(
        nx=nx,
        ny=ny,
        nvx=nvx,
        nvy=nvy,
        nu=nu,
        nv=nv,
        kq=kq,
        kr=kr,
        z=z,
    )

    plt.figure(figsize=(12, 5))

    plt.plot(x, y, label="trajectory")
    plt.plot(px1, py1, label=f"intial prediction, rms={round(rms(x, y, px1, py1), 3)}")
    print(
        f"initial rms={round(rms(x, y, px1, py1), 3)}, mean={round(mean(x, y, px1, py1), 3)}"
    )
    plt.plot(
        px2, py2, label=f"optimised prediction, rms={round(rms(x, y, px2, py2), 3)}"
    )
    print(
        f"optimised rms={round(rms(x, y, px2, py2), 3)}, mean={round(mean(x, y, px2, py2), 3)}"
    )
    plt.scatter(
        na,
        nb,
        marker="x",
        c="k",
        # label=f"noisy data, rms={round(rms(x, y, na, nb), 3)}",
    )
    print(
        f"noise rms={round(rms(x, y, na, nb), 3)}, mean={round(mean(x, y, na, nb), 3)}"
    )
    plt.legend()

    plt.title("Kalman Filter")
    plt.savefig("Report/assets/tracking/kalman_filter.png")
    # plt.show()


# 'params': {'kq': 0.01, 'kr': 0.001 + defaults}, 'rms': 2.5705388843484185}
# {'params': {'kq': 0.0175, 'kr': 0.0015 + defaults}, 'rms': 2.570039312952509}
