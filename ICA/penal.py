import autograd.numpy as np
from utils import mutual_information
from scipy.optimize import LinearConstraint, NonlinearConstraint
from scipy.sparse import csc_matrix

def get_v(A,S):
    return np.concatenate([A.T.flat, S.flat])

def get_S(v, d, l, n):
    return np.reshape(v[l*d:], (l,n))

def get_A(v, d, l, n):
    return np.reshape(v[:l*d], (d,l)).T

def get_A_col(v, j, d):
    # 0 <= j < l
    return v[j*d:(j+1)*d]

def get_A_row(v, i, d, l):
    # 0 <= i < d
    return v[i:l*d:d]

def get_source(v, j, d, l, n):
    # 0 <=j < l
    return v[l*d+j*n:l*d+(j+1)*n]

def extend_jac_source(data, j, d, l, n):
    jac = np.zeros((d+n)*l)
    jac[l*d+j*n:l*d+(j+1)*n] = data
    return jac

def single_source_zero_mean(s):
    # warning it is n times the mean
    return np.sum(s)

def single_source_zero_mean_jac(s):
    return np.ones(s.shape)

def zero_mean_eq(d, l, n):
    return [{'type': 'eq',
             'fun' : lambda v, j=j, d=d, l=l, n=n: single_source_zero_mean(get_source(v, j, d, l, n)),
             'jac' : lambda v, j=j, d=d, l=l, n=n: extend_jac_source(single_source_zero_mean_jac(get_source(v, j, d, l, n)), j, d, l, n)}
            for j in range(l)]

def lin_zero_mean_eq(d, l, n, weight=1):
    source_esperance_operator = np.zeros((l, (d+n)*l))
    for i in range(l):
        source_esperance_operator[i, d*l+i*n:d*l+(i+1)*n] = np.ones(n)
    return [LinearConstraint(csc_matrix(weight*source_esperance_operator), np.zeros(l), np.zeros(l))]

def single_source_unit_var(s):
    # diagonal term of Cov matrix S.S^T (warning it is n times the variance ...)
    return np.sum((s-np.average(s))**2)-1

def single_source_unit_var_jac(s, n):
    return 2*((n-1)/n)*(s-np.average(s))

def unit_var_eq(d, l, n):
    return [{'type': 'eq',
             'fun' : lambda v, j=j, d=d, l=l, n=n: single_source_unit_var(get_source(v, j, d, l, n)),
             'jac' : lambda v, j=j, d=d, l=l, n=n: extend_jac_source(single_source_unit_var_jac(get_source(v, j, d, l, n), n), j, d, l, n)}
            for j in range(l)]

def bilin_unit_var_eq(d, l, n, weight=1, use_hess=False):
    def fun(v):
        '''returns a (l,) array'''
        S = get_S(v, d, l, n)
        S -= np.stack(n*(np.average(S, axis=1),), axis=-1)
        return weight*np.sum(S**2, axis=1)
    
    def jac(v):
        '''returns a (l,(d+n)*l) array'''
        S = get_S(v, d, l, n)
        jac_mat = np.zeros((l, (d+n)*l))
        for i in range(l):
            S_loc = np.zeros((l,n))
            S_loc[i,:] = S[i,:]
            jac_mat[i,:] = get_v(np.zeros((d,l)), 2*(1-1/n)*(S_loc-np.stack(n*(np.average(S_loc, axis=1),), axis=-1)))
        return csc_matrix(weight*jac_mat)

    if not use_hess:
        return [NonlinearConstraint(fun, weight*np.ones(l), weight*np.ones(l), jac=jac)]
    else:
        hessians = []
        for k in range(l):
            hessian = np.zeros(((d+n)*l, (d+n)*l))
            for d_ in range(d):
                for l_ in range(l):
                    for n_1 in range(n):
                        hessian[d*l+l_*n+n_1,d*l+l_*n+n_1] += 2*(1-1/n)
                        for n_2 in range(n):
                            hessian[d*l+l_*n+n_1,d*l+l_*n+n_2] -= 2*(1-1/n)
            hessians.append(hessian)
        hessians = csc_matrix(weight*np.array(hessians))

        def hess(v,p):
            return sum([p[i]*hessians[i] for i in range(l)])

        return [NonlinearConstraint(fun, weight*np.ones(l), weight*np.ones(l), jac=jac, hess=hess)]
    
def bilin_noise_var_ineq(X, d, l, n, noise_fraction, weight=1):
    def fun(v):
        '''returns a (d*d,) array'''
        A = get_A(v, d, l, n)
        S = get_S(v, d, l, n)
        S -= np.stack(n*(np.average(S, axis=1),), axis=-1)
        AS = A.dot(S)
        noise = X-AS
        return weight*np.array(noise.dot(noise.T).flat)
    
    ones = np.ones(n)
    
    def jac(v):
        '''returns a (d*d,(d+n)*l) array'''
        A = get_A(v, d, l, n)
        S = get_S(v, d, l, n)
        S_av = np.average(S, axis=1)
        AS = A.dot(S)
        noise = X-AS
        noise_av = np.average(noise, axis=1)
        jac_mat = np.zeros((d*d, (d+n)*l))
        
        for i in range(d):
            for j in range(d):
                A_jac = np.zeros((d,l))
                A_jac[i,:] += -noise[j,:].dot(S.T) + n*noise_av[j]*S_av
                A_jac[j,:] += -noise[i,:].dot(S.T) + n*noise_av[i]*S_av
                
                S_jac = - np.outer(A[j,:], noise[i,:])     -  np.outer(A[i,:], noise[j,:]) \
                        + noise_av[i]*np.outer(A[j,:], ones)  +  noise_av[j]*np.outer(A[i,:], ones)
                
                jac_mat[i*d+j,:] = get_v(A_jac, S_jac)

        return csc_matrix(weight*jac_mat)
    print("X cov:", np.array(X.dot(X.T).flat))
    print("noise cov upper bound:", noise_fraction*np.array(X.dot(X.T).flat))
    return [NonlinearConstraint(fun, -noise_fraction*weight*abs(np.array(X.dot(X.T).flat)),
                                     +noise_fraction*weight*abs(np.array(X.dot(X.T).flat)), jac=jac)]

def noise_correlation(v, X, i, j, d, l, n, weight=1):
    A_i = get_A_row(v, i, d, l)
    A_j = get_A_row(v, j, d, l)
    S = get_S(v, d, l, n)
    AS_i =  A_i.dot(S)
    AS_j =  A_j.dot(S)
    noise_j = X[j,:] - AS_j
    cov_ij = AS_i.dot(noise_j.T) - n*np.average(AS_i)*np.average(noise_j)
    return weight*cov_ij

def noise_correlation_jac(v, X, i, j, d, l, n, weight=1):
    A_i = get_A_row(v, i, d, l)
    A_j = get_A_row(v, j, d, l)
    A_i = np.reshape(A_i, (1, l))
    A_j = np.reshape(A_j, (1, l))
    S = get_S(v, d, l, n)
    AS_i =  A_i.dot(S)
    AS_j =  A_j.dot(S)
    noise_j = X[j,:] - AS_j
    cov_ij = AS_i.dot(noise_j.T) - n*np.average(AS_i)*np.average(noise_j)

    jac_A = np.zeros((d, l))
    jac_A[i,:] += np.reshape(noise_j.dot(S.T) - np.average(noise_j)*np.sum(S.T, axis=0), (l,))
    jac_A[j,:] += np.reshape(- AS_i.dot(S.T) + np.average(AS_i)*np.sum(S.T, axis=0), (l,))

    jac_S = A_i.T.dot(noise_j) - A_j.T.dot(AS_i) + np.average(AS_i)*A_j.T.dot(np.ones((1,n))) - np.average(noise_j)*A_i.T.dot(np.ones((1,n)))

    return weight*get_v(jac_A,jac_S)

def noise_correlation_eq(X, d, l, n, weight=1):
    return [[{'type': 'eq',
             'fun' : lambda v, X=X, j=j, d=d, l=l, n=n, weight=weight: noise_correlation(v, X, i, j, d, l, n, weight=weight),
             'jac' : lambda v, X=X, j=j, d=d, l=l, n=n, weight=weight: noise_correlation_jac(v, X, i, j, d, l, n, weight=weight)}
             for j in range(d)]
            for i in range(d)]

def non_lin_noise_correlation_eq(X, d, l, n, weight=1):
    def fun(v):
        '''returns a (d*d,) array'''
        A = get_A(v, d, l, n)
        S = get_S(v, d, l, n)
        S -= np.stack(n*(np.average(S, axis=1),), axis=-1)
        AS = A.dot(S)
        return weight*np.array((AS.dot((X-AS).T)).flat)
    
    ones = np.ones(n)
    
    def jac(v):
        '''returns a (d*d,(d+n)*l) array'''
        A = get_A(v, d, l, n)
        S = get_S(v, d, l, n)
        S_av = np.average(S, axis=1)
        AS = A.dot(S)
        AS_av = A.dot(S_av)
        noise = X-AS
        noise_av = np.average(X, axis=1)-AS_av
        jac_mat = np.zeros((d*d, (d+n)*l))
        
        for i in range(d):
            for j in range(d):
                A_jac = np.ones((d,l))*AS_av[i]
                A_jac[i,:] += noise[j,:].dot(S.T) - n*noise_av[j]*S_av
                A_jac[j,:] += -AS[i,:].dot(S.T) + n*AS_av[i]*S_av
                
                S_jac =   np.outer(A[i,:], noise[j,:])     -  np.outer(A[j,:], AS[i,:]) \
                        + AS_av[i]*np.outer(A[j,:], ones)  -  noise_av[j]*np.outer(A[i,:], ones)
                
                jac_mat[i*d+j,:] = get_v(A_jac, S_jac)
        return csc_matrix(weight*jac_mat)

    return [NonlinearConstraint(fun, np.zeros(d*d), np.zeros(d*d), jac=jac)]

def source_indep_penal(d, l, n, weight=1):
    return {'type': 'penal',
            'fun' : lambda v, d=d, l=l, n=n: weight*mutual_information(get_S(v, d, l, n))}

def noise_penal(X, d, l, n, weight=1):
    penal_matrix = weight*np.ones((d,d))
    def fun(v):
        '''returns a (d*d,) array'''
        A = get_A(v, d, l, n)
        S = get_S(v, d, l, n)
        S -= np.stack(n*(np.average(S, axis=1),), axis=-1)
        AS = A.dot(S)
        noise = X-AS
        cov = noise.dot(noise.T)
        return np.sum((cov*penal_matrix)**2)
    return {'type': 'penal',
            'fun' : fun}