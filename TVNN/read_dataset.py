import numpy as np
from scipy.special import ellipj, ellipk
#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name,noise=0.0):
    if name == 'pendulum':
        return pendulum(noise)
    if name == 'lorenz':
        return lorenz(noise)

##########  Data generator  ######
def pendulum_Data(t,theta0):
    S = np.sin(0.5*(theta0) )
    K_S = ellipk(S**2)
    omega_0 = np.sqrt(9.81)
    sn,cn,dn,ph = ellipj( K_S - omega_0*t, S**2 )
    theta = 2.0*np.arcsin( S*sn )
    d_sn_du = cn*dn
    d_sn_dt = -omega_0 * d_sn_du
    d_theta_dt = 2.0*S*d_sn_dt / np.sqrt(1.0-(S*sn)**2)
    return np.stack([theta, d_theta_dt],axis=1)

def pendulum(noise, theta=2.4):

    X = pendulum_Data(np.arange(0, 9000*0.1, 0.1), theta).T
    X = X + np.random.standard_normal(X.shape) * noise
    
    Q,_ = np.linalg.qr(np.random.standard_normal((32,2)))
    X = X.T.dot(Q.T)

    return X

def  lorenzData(time=150, stepsize=0.01, N=1):
    np.random.seed(1)
    n = 3 * N
    m = round(time / stepsize)
    X = np.zeros((n, m))
    X[:, 0] = np.random.rand(1, n)
    C = 0.1
    for i in range(m - 1):
        X[0, i + 1] = X[0, i] + stepsize * (10 * (X[1, i] - X[0, i]) + C * X[0 + (N - 1) * 3, i])
        X[1, i + 1] = X[1, i] + stepsize * (20 * X[0, i] - X[1, i] - X[0, i] * X[2, i])
        X[2, i + 1] = X[2, i] + stepsize * (-8/3 * X[2, i] + X[0, i] * X[1, i])
        for j in range(1, N):
            X[0 + 3 * j, i + 1] = X[0 + 3 * j, i] + stepsize * (10 * (X[1 + 3 * j, i] - X[0 + 3 * j, i]) + C * X[0 + 3 * (j - 1), i])
            X[1 + 3 * j, i + 1] = X[1 + 3 * j, i] + stepsize * (20 * X[0 + 3 * j, i] - X[1 + 3 * j, i] - X[0 + 3 * j, i] * X[2 + 3 * j, i])
            X[2 + 3 * j, i + 1] = X[2 + 3 * j, i] + stepsize * (-8 / 3 * X[2 + 3 * j, i] + X[0 + 3 * j, i] * X[1 + 3 * j, i])
    return X.T

def lorenz(noise):

    X = lorenzData()
    X = X + 0.1*np.random.standard_normal(X.shape)*noise
    return X