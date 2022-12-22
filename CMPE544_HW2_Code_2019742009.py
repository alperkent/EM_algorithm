import numpy as np
import matplotlib.pyplot as plt

data = np.load('dataset.npy')

# define and initialize parameters
np.random.seed(0)
n = data.shape[0]
d = data.shape[1]
k = 3
threshold = 0.00000001
mu = [data[i] for i in np.random.randint(0, n, k)]
cov = [np.random.rand(d, d) for i in range(k)]
cov = [cov[i]@cov[i].T for i in range(k)]
pi = np.random.rand(k)
pi /= np.sum(pi)
log_likelihood = []
nk = np.zeros(k)
gamma = np.zeros((n, k))

# define functions
def calc_distr(i, j):
    a = pi[j]*1/((2*np.pi)**(d/2)*np.linalg.det(cov[j])**(0.5))
    b = np.exp(-0.5*(data[i]-mu[j]).T@np.linalg.pinv(cov[j])@(data[i]-mu[j]))
    return a*b

def add_like():
    log_like = 0
    for i in range(n):
        inner_sum = 0
        for j in range(k):
            inner_sum += calc_distr(i, j)
        log_like += np.log(inner_sum)
    log_likelihood.append(log_like)

def calc_gamma():
    new_gamma = np.zeros((n, k))
    for i in range(n):
        denom = 0
        for j in range(k):
            denom += calc_distr(i, j)
        for j in range(k):
            new_gamma[i][j] = calc_distr(i, j)/denom
    return new_gamma

def calc_nk():
    for i in range(k):
        nk[i] = np.sum(gamma[:, i])

def get_mu():
    mu_new = np.zeros((k, d))
    for i in range(k):
        for j in range(n):
            mu_new[i] += gamma[j][i]*data[j]
        mu_new[i] /= nk[i]
    return mu_new

def get_cov():
    cov_new = [np.zeros((d, d)) for i in range(k)]
    for i in range(k):
        for j in range(n):
            cov_new[i] += gamma[j][i]*np.outer((data[j]-mu[i]), (data[j]-mu[i]).T)
        cov_new[i] /= nk[i]
    return cov_new

def calc_pi():
    for i in range(k):
        pi[i] = nk[i]/n

def get_diff():
    return log_likelihood[-1] - log_likelihood[-2]

# run the algorithm
add_like()
gamma = calc_gamma()
calc_nk()
mu = get_mu()
cov = get_cov()
calc_pi()
add_like()
diff = get_diff()
while np.abs(diff) > threshold:
    gamma = calc_gamma()
    calc_nk()
    mu = get_mu()
    cov = get_cov()
    calc_pi()
    add_like()
    diff = get_diff()

# plot clusters
labels = np.where(gamma[:] > 0.9)[1]
plt.scatter(data[np.where(labels == 0), 0], data[np.where(labels == 0), 1], c='blue', marker='.')
plt.scatter(data[np.where(labels == 1), 0], data[np.where(labels == 1), 1], c='red', marker='.')
plt.scatter(data[np.where(labels == 2), 0], data[np.where(labels == 2), 1], c='green', marker='.')
for i in range(k):
    plt.scatter(mu[i][0], mu[i][1], marker='x', c='black', s=100)
plt.show()