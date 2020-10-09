import numpy as np
import matplotlib.pyplot as plt

def kf_predict(X, P, A, Q, B, U):
    X = np.dot(A, X) + np.dot(B, U)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return X, P


def kf_update(X, P, Y, H, R):
    IM = np.dot(H, X)
    IS = R + np.dot(H, np.dot(P, H.T))
    K = np.dot(P, np.dot(H.T, np.linalg.inv(IS)))
    X = X + np.dot(K, (Y - IM))
    LH = gauss_pdf(Y, IM, IS)
    print(LH)
    return X, P, K, IM, IS, LH


def gauss_pdf(X, M, S):
    if np.shape(M)[1] == 1:
        DX = X - np.tile(M, np.shape(X)[1])
        E = 0.5 * np.sum(DX * np.dot(np.linalg.inv(S), DX), axis=0)
        E = E + 0.5 * np.shape(M)[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    elif np.shape(X)[1] == 1:
        DX = np.tile(X, M.shape()[1]) - M
        E = 0.5 * np.sum(DX * np.dot(np.linalg.inv(S), DX), axis=0)
        E = E + 0.5 * np.shape(M)[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)
    else:
        DX = X - M
        E = 0.5 * np.sum(DX * np.dot(np.linalg.inv(S), DX))
        E = E + 0.5 * np.shape(M)[0] * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(S))
        P = np.exp(-E)

    return P[0], E[0]


# initialize the parameters
# Initialize the state matrices
dt = 0.1
X = np.array([[0], [0], [0.1], [0.1]])
print(np.shape(X))
P = np.diag((0.01, 0.01, 0.01, 0.01))
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q = np.eye(np.shape(X)[0])
B = np.eye(np.shape(X)[0])
U = np.zeros((np.shape(X)[0], 1))
# Initialize the measurement matrices
Y = np.array([[X[0, 0] + np.abs(np.random.randn(1)[0])], [X[1, 0] + np.abs(np.random.randn(1)[0])]])
print(Y)
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = np.eye(np.shape(Y)[0])

# Number of iterations in Kalman Filter
N_iter = 50

# Apply the Kalman Filter

i = 0
x_pred = []
x_update = []
y = []
print("At time step",i/10," The state is ")
print(X)
print("At time step",i/10," The measurement is ")
print(Y)
print("**************************************")
# Apply the Kalman Filter
for i in np.arange(0, N_iter):
    X, P = kf_predict(X, P, A, Q, B, U)
    print("At time step",i/10," The predicted state is ")
    print(X)
    x_pred.append(X[0:2])
    X, P, K, IM, IS, LH = kf_update(X, P, Y, H, R)
    print("At time step",(i+1)/10," The updated state is ")
    print(X)
    x_update.append(X[0:2])
    Y = np.array([[X[0, 0] + np.abs(0.1 * np.random.randn(1)[0])],
                  [X[0, 0] + np.abs(0.1 * np.random.randn(1)[0])]])
    print("At time step",(i+1)/10," The measurement is ")
    print(Y)
    y.append(Y)
    print("**************************************")

x_pred = np.array(x_pred).reshape(50,2)
x_update = np.array(x_update).reshape(50,2)
y = np.array(y).reshape(50,2)

x_pred_plot = plt.plot(x_pred[:,0],x_pred[:,1],color='k', linestyle = '-', label='predicted states')
x_update_plot = plt.plot(x_update[:,0],x_update[:,1],color='b',linestyle = '-',label='updated states')
y_plot = plt.plot(y[:,0],y[:,1],color = 'r',linestyle = ':',label='measurement')
plt.axis([0,4,0,4])
plt.legend()
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.title("Tracking vehicle with Kalman filter")
plt.show()