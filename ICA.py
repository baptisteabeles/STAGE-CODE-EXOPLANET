import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def entropy(S):
    S = np.array(random.sample(list(S.T[::]), n)).T 
    d, n = np.shape(S)
    lambdas = np.zeros(n)
    for i in range(n):
        v = np.stack(n*(S[:, i],), axis=-1)
        dists = np.sum((v-S)**2, axis=1)
        lambdas[i] = np.min(dists[dists>0])
    return d*np.mean(np.log2(lambdas)) #+ np.log2((n-1)/d) + np.euler_gamma/np.log(2) + np.log2(d*np.pi**(d/2)/gamma(1+d/2))

def plot_2D_distrib(X, title='', plot_entropy=False):
    '''Plot the data X, assuming np.shape(X) is of form (2,n)'''
    
    fig, ax = plt.subplots()
    plt.scatter(X[0,:], X[1,:], marker='x', s=20, color='purple')
    plt.xlabel('first data dimension')
    plt.ylabel('second data dimension')
    plt.title('2D view of the data '+title, fontsize=20)
    plt.axis('equal')
    plt.grid(True)
    plt.figtext(x=0.6,y=0.8,s='$H={:.3g}$'.format(entropy(X)), fontsize=20) if plot_entropy else None
    
    return ax


def find_column(A):
    var_list = list(map(lambda col : sum((col-np.mean(col))**2), A.T[::]))
    return np.argmin(var_list), min(var_list)

def update(A, on_place=False, eps=1e-6, verbose=False):
    A = np.copy(A) if not on_place else A
    i, regularity = find_column(A)
    print('column targeted: ', A[:,i]) if verbose else None
    print('regularity: ', regularity) if verbose else None
    A[:,i] = np.mean(A[:,i]) if regularity>eps else A[:,i]
    return A if not on_place else regularity>eps

def prepare_const_column(A_hat):
    A_hat_const = np.copy(A_hat)
    i, _ = find_column(A_hat)
    col_0 = np.copy(A_hat_const[:, 0])
    A_hat_const[:, 0] = A_hat_const[:, i]
    A_hat_const[:, i] = col_0
    return A_hat_const


def ICA(X, method_independence='FOBI', constraint=None, verbose=False, plot_entropy=False):
    '''Input: - X: a (d, n) array.
              - method_independence: string that specifies the method \\
                used to optimze statistical independence of the output S.
       Returns: W (d, d), A_hat (d, d) and S (d, n) such that: \\
                 $$ X = A_hat . S $$ \\
                 $$ A_hat = W^{-1} $$\\
                 With S whitened and having a maximized statistical independence.'''
        
    d, n = np.shape(X)
    
    # substract off the mean for each dimension
    X_av = np.reshape(np.average(X, axis=1), (d,1))
    X -= X_av   
    if verbose:
        ax = plot_2D_distrib(X, title='centered', plot_entropy=False)
        plt.show()
    
    RY_2 = X.dot(X.T)      # compute the covariance : 2nd order moment
    eigs_RY_2, U = np.linalg.eig(RY_2)    # RY_2 = U.dot(np.diag(eigs_RY_2).dot(U.T))
    
    X_w = np.diag(eigs_RY_2**(-1/2)).dot(U.T).dot(X)   # whiten the data
    if verbose:
        ax = plot_2D_distrib(X_w, title='whitened', plot_entropy=plot_entropy)
        plt.show()
        
    
    if method_independence == 'FOBI':
        # compute the 4th order moment following FOBI method as depicted in [2]
        RY_4 = X_w.dot((X_w.T.dot(X_w)*np.eye(n)/n).dot(X_w.T))

        eigs_RY_4, V_T = np.linalg.eig(RY_4)    # extract the eigenvectors of RY_4 to get V
        V = V_T.T
        print('Eigenvalues in FOBI: '+str(eigs_RY_4)) if verbose else None
    
    if np.sum(np.abs(V.dot(V.T)-np.eye(d))**2)>1e-4:
        raise Exception('Matrix V is not orthogonal: \n'+str(V))

    W = V.dot(np.diag(eigs_RY_2**(-1/2)).dot(U.T))     # S_hat = W x X
    A_hat = U.dot(np.diag(eigs_RY_2**(1/2)).dot(V.T))  # X = A_hat x S_hat, A_hat = W^{-1}
    S_hat = W.dot(X+X_av)
    
    if verbose:
        ax = plot_2D_distrib(S_hat, title='estimated sources', plot_entropy=plot_entropy)
        plt.show()
               
    if constraint=='const-column':
        
        print("FOBI A_hat \n", A_hat/A_hat[0,0]) if verbose else None
        A_hat = prepare_const_column(A_hat)
        print("re-organized FOBI A_hat \n", A_hat/A_hat[0,0]) if verbose else None
        
        K = U.dot(np.diag(eigs_RY_2**(-1))).dot(U.T)
        assert np.sum(K) > 0
        
        target = U.T.dot(A_hat)
        
        penal = lambda x: np.sum((np.array(target.T.flat)-x)**2)
        penal_der = lambda x: 2*(x-np.array(target.T.flat))
        
        c = + (np.sum(K))**-0.5
        A_hat_c = np.average(A_hat[:,0])
        print("A_hat const-column average: ", A_hat_c) if verbose else None
        if abs(c-A_hat_c)>abs(-c-A_hat_c):
            c *= -1
        print("c =",c) if verbose else None

        col = np.zeros((d,d))
        col[:,0] = c*U.T.dot(np.ones(d))
        
        def eq_cons_jac(x):
            jac = 0*x
            jac[0:d] = 2*(x[0:d]-np.array(col.T.flat[0:d]))
            return jac
    

        eq_cons = {'type': 'eq',
                   'fun' : lambda x: np.sum((x[0:d]-np.array(col.T.flat[0:d]))**2),
                   'jac' : eq_cons_jac}

        def D_cons_fun(x):
            alpha = np.reshape(x, (d,d)).T
            return np.sum((alpha.T.dot(np.diag(eigs_RY_2**(-1)).dot(alpha)) - np.eye(d))**2)

        def D_cons_jac(x):
            alpha = np.reshape(x, (d,d)).T
            jac = 4*np.diag(eigs_RY_2**(-1)).dot(alpha.dot(alpha.T.dot(np.diag(eigs_RY_2**(-1)).dot(alpha)) - np.eye(d)))
            return np.array(jac.T.flat)

        D_cons = {'type': 'eq',
                  'fun' : D_cons_fun,
                  'jac' : D_cons_jac}
        
        res = minimize(penal, 
                       x0=np.array(target.T.flat),
                       method='SLSQP',
                       jac=penal_der,
                       constraints=[eq_cons, D_cons], 
                       options={'ftol': 1e-9, 'disp': verbose, 'maxiter': 200})
        
        alpha = np.reshape(res.x, (d,d)).T
        A_hat = U.dot(alpha)
        print("closest const-col A_hat found:\n", A_hat/c) if verbose else None

        W = np.linalg.inv(A_hat)
        S_hat = W.dot(X)

        if verbose:
            ax = plot_2D_distrib(S_hat, title='final estimated sources', plot_entropy=plot_entropy)
            plt.show()
    
    return W, A_hat, S_hat