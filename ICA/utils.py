import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors


def plot_2D_distrib(X, title='', loss=None, loss_name=None):
    '''Plot the data X, assuming np.shape(X) is of form (2,n)'''
    if X.shape[0]>2:
        return
    fig, ax = plt.subplots()
    plt.scatter(X[0,:], X[1,:], marker='x', s=20, color='purple')
    plt.xlabel('first data dimension', fontsize=15)
    plt.ylabel('second data dimension', fontsize=15)
    plt.title('2D view of the data '+title, fontsize=20)
    plt.axis('equal')
    plt.grid(True)
    plt.figtext(x=0.6,y=0.8,s='$'+loss_name+'={:.3g}$'.format(loss(X)), fontsize=20) if loss is not None else None
    
    return ax

def plot_comparison(x, y, xlabel=None, ylabel=None):
    fig = plt.figure()
    plt.scatter(x, y, label=r'$R^2={:.4f}$'.format(np.corrcoef(x,y)[0,1]), marker='x')
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(ylabel+' against '+xlabel, size=20)
    return fig

def plot_A(A):
    fig = plt.figure()
    plt.imshow(A)
    plt.title('Matrix A, graded:'+str(linear_entropy(A)))
    return fig

def mean(X):
    '''Returns: X (d,n) and X_av (d,) such that 
                X is finally zero meaned over time (axis 1)'''
    
    d, _ = X.shape
    # substract off the mean for each dimension
    X_av = np.average(X, axis=1)
    return X-np.reshape(X_av, (d,1)), X_av

def whiten(X, sort=True):
    '''Asserts X is zero meaned.
       Returns: X_w (d,n), Whiten_mat (d,d) and Whiten_mat_inv (d,d) such that
                with X_w.dot(X_w.T)=np.eye(d) and Whiten_mat.dot(X)=X_w'''
    
    RY_2 = X.dot(X.T)      # compute the covariance : 2nd order moment
    eigs_RY_2, U = np.linalg.eig(RY_2)    # RY_2 = U.dot(np.diag(eigs_RY_2).dot(U.T))
    if sort:
        inds = (-eigs_RY_2).argsort()
        eigs_RY_2 = eigs_RY_2[inds]
        U = U[:, inds]
    Whiten_mat = np.diag(eigs_RY_2**(-1/2)).dot(U.T)
    Whiten_mat_inv = U.dot(np.diag(eigs_RY_2**(1/2)))
    X_w = Whiten_mat.dot(X)   # whiten the data
    return X_w, Whiten_mat, Whiten_mat_inv, eigs_RY_2

def check_orthogonal(V, eps=1e-5):
    d, _ = V.shape
    if np.sum(np.abs(V.dot(V.T)-np.eye(d))**2)>eps:
        raise Exception('Matrix V is not orthogonal: \n'+str(V))
    else:
        return
    
def standardize(signal):
    avg = np.average(signal)
    return (signal-avg)/(signal-avg).dot((signal-avg).T)**.5
    
def SNR(true_signal, estimation):
    return np.average((true_signal-np.average(true_signal))**2)/np.average((true_signal-estimation-np.average(true_signal-estimation))**2)

def analyze(X, num_sources, noise_fraction, verbose, plot_loss, plot_loss_name, result_traceback, eps=1e-5, **kwargs):
    '''Raises exception in case of an incoherent result.
       Prints the results if verbose is set to True.'''
    d, n = X.shape
    
    results = result_traceback[-1]
    S = results['S']
    l, n = S.shape
    assert l==num_sources
    
    S, S_av = mean(S)
    if np.max(abs(S_av))>eps:
        raise Exception('Sources S not zero meaned:\n'+str(S_av))  

    if np.max(abs(np.diag(S.dot(S.T))-np.ones(l)))>eps:
        raise Exception('Sources S variance not unitary:\n'+str(S.dot(S.T)))

    A = results['A']
    
    noise = X-A.dot(S)
    noise, noise_av = mean(noise)
    if np.max(abs(noise_av))>eps:
        raise Exception('Noise is not zero meaned:\n'+str(noise_av))

    if np.any((noise_fraction+eps)*abs(np.diag(X.dot(X.T)))<abs(np.diag(noise.dot(noise.T)))):
        raise Exception('Noise co-variance too high:\n',
                        noise_fraction, X.dot(X.T), noise.dot(noise.T))
    if verbose:
        print("A:\n",A)
        plot_A(A)
        print('S average: ', S_av)  
        print('noise average: ', noise_av)
        print('S cov:\n', S.dot(S.T))
        print(plot_loss_name+'(S):', plot_loss(S))
        print('noise cov:\n', noise.dot(noise.T))
        print('X cov:\n', X.dot(X.T))
        print('cov(AS,noise):\n', noise.dot((A.dot(S)).T))
        ax = plot_2D_distrib(S, title='current estimated sources', loss=plot_loss, loss_name=plot_loss_name)
        plt.show()
        if np.max(noise.dot(noise.T))>eps:
            ax = plot_2D_distrib(noise, title='current estimated noise', loss=plot_loss, loss_name=plot_loss_name)
            plt.show()
        


# Usefull code for stats calculus extracted from :
# https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
# adapted to be used with our data format

def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_features, n_samples)
        The data the entropy of which is computed
        If X is a one-dimension array, X is reshaped as (1, n_samples)
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    if len(X.shape)==1:
        X = np.reshape(X, (1, X.shape[0]))
    
    # Distance to kth nearest neighbor
    r = nearest_distances(X.T, k) # squared distances
    l, n = X.shape                # l is the number of sources, n the number of samples
    volume_unit_ball = (np.pi**(.5*l)) / gamma(.5*l + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return l*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (l*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def mutual_information(variables, k=1):
    '''
    Returns the mutual information between any number of variables.
    Each variable is a vector (n_samples,) or a matrix X = array(n_features, n_samples)
    where
      n = number of samples
      dx,dy = number of dimensions
    
    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation
    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    '''
    if len(variables) < 2:
        raise AttributeError("Mutual information must involve at least 2 variables")
    all_vars = np.vstack(variables)
    return (sum([entropy(X, k=k) for X in variables])
            - entropy(all_vars, k=k))



# personal methods, less efficient

def source_entropy(S, eps=0):
    '''Implemented entropy calculus with using an estimator as depicted in [3] eq.10.'''
    if len(S.shape)==1:
        S = np.reshape(S, (1, S.shape[0]))
    l, n = S.shape
    lambdas_sq = np.zeros(n)
    for i in range(n):
        v = np.stack(n*(S[:,i],), axis=-1)
        dists = np.sum((v-S)**2, axis=0)
        lambdas_sq[i] = max(np.min(dists[dists>0]), eps**2)
    return l*np.average(np.log2(lambdas_sq))/2 + np.log2((n-1)/l) + \
           np.euler_gamma/np.log(2) + np.log2(l*(np.pi**(l/2))/gamma(1+l/2))

def multi_information(S):
    '''I(S) = \sum_{i=1}^l H(S_i) - H(S)'''
    return sum(list(map(source_entropy, S)))-source_entropy(S)

def linear_entropy(A_hat, plot=False):
    ''' Asses how matrix A_hat is likely to operate with over a source S \\
        containing only one source equally distributed over measures. \\
        More explicitely, A_hat should have only one column almost constant \\
        while other columns should be zero meaned. Hence we use the entropy \\
        formula over columns of A_hat as a grade for the matrix:
        
        Input: - A_hat: a (d, d) array.
        Returns: \sum_{j=1}^{d} -p_j*ln(p_j) with p_j = s_j^2/\sum_i s_j^2 with s_j^2 = \sum_i (A_hat_ij-\bar{A_hat_j})^2
        '''
    d, l = A_hat.shape
    col_avgs = np.sum(A_hat, axis=0)/d
    s_2 = np.sum((A_hat-col_avgs)**2, axis=0)/col_avgs**2
    s_2 /= np.sum(s_2)
    probs = thresholder(s_2)
    probs /= np.sum(probs)
    probs_light = probs[probs.nonzero()]
    if plot:
        plt.figure()
        plt.xlabel('column number', fontsize=15)
        plt.ylabel('column variance', fontsize=15)
        plt.scatter(range(l), s_2, marker='x')
        #plt.scatter(range(l), np.sum((A_hat)**2, axis=0), marker='+', label='amp')
        #plt.legend()
        plt.grid(True)
        plt.title('Matrix A entropy: {:.3g}'.format(np.sum(-probs_light*np.log(probs_light))), fontsize=20)
        plt.show()
    return np.sum(-probs_light*np.log(probs_light))

def thresholder(x, threshold=0.3):
    return np.exp(-(x/threshold)**2)

def approx_fgrad(x, f, eps=1e-1, centered=False):
    
    delta_x = eps*np.eye(len(x.flat))
    
    if centered:
        grad = list(map(lambda delta: (f(x+np.reshape(delta/2, x.shape)) - f(x-np.reshape(delta/2, x.shape)))/eps, delta_x))
    else:
        y = f(x)
        grad = list(map(lambda delta: (f(x+np.reshape(delta, x.shape)) - y)/eps, delta_x))

    grad = np.asarray(grad)
        
    if len(x.shape)>1:
        grad = np.reshape(grad, x.shape+grad.shape[1:])

    return grad