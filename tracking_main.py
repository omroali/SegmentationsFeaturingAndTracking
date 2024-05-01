
from matplotlib import pyplot as plt
import numpy as np

def kalman_predict(x, P, F, Q):
    xp = F*x
    Pp = F*P*F.T + Q
    return xp, Pp

def kalman_update(x, P, H, R, z):
    S = H*P*H.T + R
    K = P*H.T*np.linalg.inv(S)
    zp = H*x

    xe = x + K*(z - zp)
    Pe = P - K*H*P
    return xe, Pe

def kalman_tracking(z, dt = 0.5, kq = 0.16, kr = 0.25):
    # Constant Velocity
    F = np.matrix([[1, dt, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dt],
                    [0, 0, 0, 1]])

    # Cartesian observation model
    H = np.matrix([[1, 0, 0, 0],
                   [0, 0, 1, 0]])
    

    Q = kq*np.matrix([[1, 0, 0, 0],
                   [0, 2, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 2]])
    
    R = kr*np.matrix([[1, 0],
                   [0, 1]])

    x = np.matrix([0, 0, 0, 0]).T
    P = Q

    N = len(z[0])
    s = np.zeros((4,N)) 

    for i in range(N):
        xp, Pp = kalman_predict(x, P, F, Q)
        x, P = kalman_update(xp, Pp, H, R, z[:,i])
        val = np.array(x[:2,:2]).flatten()
        s[:,i] = val
    
    px = s[0,:]
    py = s[1,:]

    return px, py

if __name__ == "__main__":

    x  = np.genfromtxt('data/x.csv',  delimiter=',')
    y  = np.genfromtxt('data/y.csv',  delimiter=',')
    na = np.genfromtxt('data/na.csv', delimiter=',')
    nb = np.genfromtxt('data/nb.csv', delimiter=',')
    z = np.stack((na, nb))
    plt.plot(x, y)
    px, py = kalman_tracking(z)
    lg = ['True', 'Kalman']
    for kr in [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        for kq in [0.01]:
                #    , 0.1 , 1, 10, 100, 1000, 10000]:
            px, py = kalman_tracking(z, dt = 0.5, kr = kr, kq = kq)
            plt.plot(px, py)
            lg.append('kr = ' + str(kr) + ', kq = ' + str(kq))
    
    plt.legend(lg)

    # for dt in [0.5]:
    #     # for kq in 10*[0.16, 0.14, 0.12, 0.10]:
    #         for kr in 10*[0.25, 0.20, 0.15, 0.10]:
    #             px, py = kalman_tracking(z, dt = dt, kr = kr)
    #             plt.plot(px, py)
    plt.scatter(na, nb)
    plt.show()

    import optuna

    def objective(trial):
        kr = trial.suggest_uniform('kr', 0.0, 1.0)
        kq = trial.suggest_uniform('kq', 0.0, 1.0)

        # Replace this with your actual Kalman filter code
        # and a metric that measures its performance
        # For example, you could run the filter on some test data
        # and compute the mean squared error of the estimates
        mse = run_kalman_filter(kr, kq)

        return mse