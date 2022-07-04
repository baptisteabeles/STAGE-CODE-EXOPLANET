import autograd.numpy as np
import numdifftools as nd
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from utils import mean, whiten, mutual_information, approx_fgrad, check_orthogonal, linear_entropy

from penal import source_indep_penal, noise_penal, bilin_unit_var_eq, lin_zero_mean_eq, \
                  non_lin_noise_correlation_eq, bilin_noise_var_ineq, get_v, get_A, get_S

from scipy.optimize import shgo, differential_evolution, basinhopping

from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import SteepestDescent, ParticleSwarm, NelderMead, TrustRegions, ConjugateGradient

from sklearn.decomposition import FastICA
from sklearn.utils.validation import check_random_state
from sklearn.decomposition._fastica import _sym_decorrelation

from coroica import CoroICA

from JADE import jadeR

def sk_FastICA(X, 
               num_sources, 
               max_iter=1500, 
               algorithm='deflation', 
               **kwargs):
    transformer = FastICA(n_components=num_sources,
                          max_iter=max_iter,
                          algorithm=algorithm)
    S = transformer.fit_transform(X.T).T
    A = transformer.mixing_
    results = {'S': S, 'A': A}
    return results 

def coro_ICA(num_sources, 
             schedule, 
             result_traceback, 
             max_iter=1500, 
             algorithm='deflation', 
             **kwargs):
    '''Asserts PCA heuristic has been run just before.'''
    assert str(list(map(lambda method: method.__name__, [PCA, coro_ICA]))).strip('[]') in \
           str(list(map(lambda method: method.__name__, schedule))).strip('[]')
    transformer = CoroICA(n_components=num_sources,
                          max_iter=max_iter,)
    last_results = result_traceback[-1]
    X_w = last_results['S']
    _, n = X_w.shape
    S = (n**-.5)*transformer.fit_transform((n**.5)*X_w.T).T
    W = transformer.V_
    W_pinv = np.linalg.pinv(W)
    A = last_results['A'].dot(W_pinv)  
    results = {'S': S, 'A': A}
    return results 

def PCA(X_w, 
        num_sources, 
        Whiten_mat_inv, 
        result_traceback, 
        **kwargs):
    A = Whiten_mat_inv[:,:num_sources]
    S = X_w[:num_sources,:]
    results = {'S': S, 'A': A}
    return results

def partial_FastICA(X_w,
                    noise_add,
                    Whiten_mat,
                    Whiten_mat_inv,
                    num_sources, 
                    schedule, 
                    constraints, 
                    result_traceback, 
                    max_iter=1000, 
                    algorithm='parallel', 
                    random_state=None, 
                    **kwargs):
    '''Designed to use sk_FastICA in const column case.
       Asserts PCA has been run just before during pre processing and X_w is sorted with decreasing power.'''
    
    X_w_loc = np.copy(X_w[:num_sources,:])

    random_state = check_random_state(random_state)
    W_init = np.array(random_state.normal(size=(num_sources, num_sources)), dtype=X_w_loc.dtype)
    if noise_add>0:
        _, n = X_w_loc.shape
        ############### attention normalment il faut appliquer whiten_mat au bruit !
        X_w_loc += Whiten_mat[:num_sources,:num_sources].dot(noise_add*mean(np.array(random_state.normal(size=(num_sources, n)), dtype=X_w_loc.dtype))[0])
        #X_w_loc += noise_add*mean(np.array(random_state.normal(size=(num_sources, n)), dtype=X_w_loc.dtype))[0]
        X_w_loc = _sym_decorrelation(X_w_loc)
        assert np.max(np.abs(X_w_loc.dot(X_w_loc.T)-np.eye(num_sources)))<1e-10
    
    transformer = FastICA(whiten=False, 
                          max_iter=max_iter, 
                          w_init=W_init,
                          algorithm=algorithm)
    
    S = transformer.fit_transform(X_w_loc.T).T
    V_T = transformer.mixing_
    # change the cropped PCA matrix (n_features, n_components) using V (n_components, n_components)
    A = Whiten_mat_inv[:,:num_sources]
    A[:num_sources,:] *= 0
    A = A.dot(V_T)    
    
    results = {'S': S, 'A': A}
    del X_w_loc
    return results 


def FOBI(X_w, 
         verbose, 
         Whiten_mat_inv, 
         result_traceback, 
         **kwargs):
    '''Asserts X_w is zero meaned and whiten.
       Returns: V (d,d) and S:=V^{-1}.dot(X_w) (d,n)
                with V an orthogonal matrix computed according to 
                the 4th order moment method FOBI, as depicted in [2]'''
    d, n = X_w.shape
    RY_4 = X_w.dot((X_w.T.dot(X_w)*np.eye(n)/n).dot(X_w.T))
    eigs_RY_4, V_T = np.linalg.eig(RY_4)    # extract the eigenvectors of RY_4 to get V.T
    V = V_T.T
    
    print('Eigenvalues in FOBI: '+str(eigs_RY_4)) if verbose else None
    
    check_orthogonal(V)
    
    A = Whiten_mat_inv.dot(V.T)
    results = {'V': V, 'S': V.dot(X_w), 'A': A}
    return results

def noise_optim(X, 
                num_sources, 
                noise_fraction, 
                verbose, 
                result_traceback, 
                penal_weights=[1,1], 
                const_weights=[1,1,1,1], 
                optim_type='penal', 
                **kwargs):
    '''optim_type: if 'ineq', the noise covariance amplitude is bounded as a constraint.
                   if 'penal', the noise covariance is added as a penalisation term'''
    d, n = X.shape
    
    # num_sources is the number of sources to find so that :
    #     np.shape(A) = d, num_sources
    #     np.shape(S) = num_sources, n
    # v[:d*num_sources] is for A, v[d*num_sources:] is for S
    # A.T.flat = v[:d*num_sources]
    
    last_results = result_traceback[-1]
    
    A0 = last_results['A']
    S0 = last_results['S']
    v0 = get_v(A0,S0)
    method = 'trust-constr'
    options = {'disp': verbose, 'maxiter': 20}
    
    constraints = lin_zero_mean_eq(d, num_sources, n, weight=const_weights[0])+\
                  bilin_unit_var_eq(d, num_sources, n, weight=const_weights[1]) +\
                  non_lin_noise_correlation_eq(X, d, num_sources, n, weight=const_weights[2])
    
    if optim_type=='ineq':
        penal = lambda v: source_indep_penal(d, num_sources, n, weight=penal_weights[0])['fun'](v)
        constraints += bilin_noise_var_ineq(X, d, num_sources, n, noise_fraction=noise_fraction, weight=const_weights[3])
        res = minimize(penal, 
                       x0=v0,
                       constraints=constraints,
                       options=options,
                       method=method)
        
    elif optim_type=='penal':
        penal = lambda v: source_indep_penal(d, num_sources, n, weight=penal_weights[0])['fun'](v) + \
                          noise_penal(X, d, num_sources, n, weight=penal_weights[1])['fun'](v)
        res = minimize(penal, 
                       x0=v0,
                       constraints=lin_zero_mean_eq(d, num_sources, n, weight=const_weights[0])+\
                                   bilin_unit_var_eq(d, num_sources, n, weight=const_weights[1]) +\
                                   non_lin_noise_correlation_eq(X, d, num_sources, n, weight=const_weights[2]),
                       options=options,
                       method=method)
        
    print(res) if verbose else None    
    
    A = get_A(res.x, d, num_sources, n)
    S = get_S(res.x, d, num_sources, n)
    
    print('||A-A_0||^2=', np.sum((A-A0)**2))
    
    results = {'S': S, 'A': A}
    
    return results

def V_optim(X_w, 
            num_sources, 
            verbose, 
            constraints, 
            Whiten_mat_inv, 
            result_traceback, 
            eps=1e-4, 
            **kwargs):

    entropy_constraint = 'less-entropy' in constraints

    d, n = X.shape
    assert num_sources==d, "Case num_sources!=d not yet implemented"
    
    try:
        last_results = result_traceback[-1]
        V0 = last_results['V']
    except:
        print('Failed recovering V from last computation, initializing with identity instead...')
        V0 = np.eye(d)
        
    def vectorize(V):
        return np.array(V.T.flat)
    
    def unvectorize(v):
        return np.reshape(v, (d,d)).T
    
    def V_orthogonality_eq(d):
        def fun(v):
            V = unvectorize(v)
            return np.sum((V.dot(V.T)-np.eye(d))**2)
        #return [NonlinearConstraint(fun, vectorize(np.zeros((d,d))), vectorize(np.zeros((d,d))))]
        return [{'type': 'eq',
                 'fun': fun}]
    
    V_constraints = V_orthogonality_eq(d)
    #method = 'trust-constr'
    options ={'disp': verbose, 'maxiter': 200}  # maybe add ftol=...
    def V_penal(v):
        V = unvectorize(v)
        S = V.dot(X_w)
        return 1e5*mutual_information(S)+1e10*np.sum((V.dot(V.T)-np.eye(d))**2)
    
    res = minimize(V_penal, 
                   x0=vectorize(V0),
                   constraints=[], #V_constraints, 
                   #method=method,
                   options=options)
    
    V = unvectorize(res.x)
    check_orthogonal(V)
        
    print('||V-V_0||^2=', np.sum((V-V0)**2))
    
    A = Whiten_mat_inv.dot(V.T)
    results = {'V': V, 'S': V.dot(X_w), 'A': A}
    
    return results

def V_global_optim(X_w, 
                   num_sources, 
                   verbose, 
                   constraints, 
                   Whiten_mat_inv, 
                   result_traceback, 
                   eps=1e-4, 
                   optim_type=differential_evolution, 
                   **kwargs):

    entropy_constraint = 'less-entropy' in constraints

    d, n = X.shape
    assert num_sources==d, "Case num_sources!=d not yet implemented"
    
    try:
        last_results = result_traceback[-1]
        V0 = last_results['V']
    except:
        print('Failed recovering V from last computation, initializing with identity instead...')
        V0 = np.eye(d)
        
    def vectorize(V):
        return np.array(V.T.flat)
    
    def unvectorize(v):
        return np.reshape(v, (d,d)).T
        
    if optim_type.__name__=='basinhopping':
        
        orth_const_penal = 1e8
        
        def V_penal(v):
            V = unvectorize(v)
            S = V.dot(X_w)
            return mutual_information(S)+orth_const_penal*np.sum((V.dot(V.T)-np.eye(d))**2)
        
        res = optim_type(V_penal, 
                         x0=vectorize(V0),
                         disp=verbose)
        
    else:
        def V_orthogonality_eq(d, optim_type=optim_type):
            def fun(v):
                V = unvectorize(v)
                return vectorize(V.dot(V.T)-np.eye(d))
            if optim_type.__name__=='shgo':
                return [{'type': 'eq', 'fun': fun}]
            elif optim_type.__name__=='differential_evolution':
                return [NonlinearConstraint(fun, np.zeros(d**2), np.zeros(d**2))]

        bounds = (d**2)*[(-1,1)]
        V_constraints = V_orthogonality_eq(d)
        options = {'disp': verbose}
        
        def V_penal(v):
            V = unvectorize(v)
            S = V.dot(X_w)
            return mutual_information(S)
        
        res = optim_type(V_penal, 
                         constraints=V_constraints, 
                         bounds=bounds, 
                         #options=options,
                         disp=verbose,)
    
    V = unvectorize(res.x)
    check_orthogonal(V)
        
    print('||V-V_0||^2=', np.sum((V-V0)**2))
    
    A = Whiten_mat_inv.dot(V.T)
    results = {'V': V, 'S': V.dot(X_w), 'A': A}
    
    return results

def A_manifold_const(X_w,
                     num_sources, 
                     verbose,
                     constraints, 
                     Whiten_mat, 
                     Whiten_mat_inv,
                     result_traceback, 
                     **kwargs):
    assert 'less-entropy' in constraints

    try:
        last_results = result_traceback[-1]
        V0 = last_results['V']
    except:
        try:
            V0 = Whiten_mat.dot(last_results['A']).T
        except:
            print('Failed recovering V from last computation, initializing with identity instead...')
            V0 = np.eye(d)
            assert num_sources==d, "Case num_sources!=d not yet implemented"
                
    d, l = V0.shape
        
    # (1) Instantiate a manifold
    if d==l:
        if np.linalg.det(V0)<0:
            print('Negative determinant for V0:', np.linalg.det(V0)) if verbose else None
            V0[:,0] *= 1
        manifold = Rotations(d, k=1)
    else:
        manifold = Stiefel(d, l)
    ############################# Question : should we use only rotations to go faster ?!
    
    def cost(V):
        return linear_entropy(np.dot(Whiten_mat_inv, V.T))
    
    problem = Problem(manifold=manifold,
                      cost=cost,
                      verbosity=2*int(verbose))

    # (3) Instantiate a Pymanopt solver
    # ConjugateGradient is way better than SteepestDescent
    solver = ConjugateGradient(logverbosity=2*int(verbose))

    # let Pymanopt do the rest
    if verbose:
        V, loginfo = solver.solve(problem, x=V0)
        print('Cost(V)=', cost(V))
        print('||V-V_0||^2=', np.sum((V-V0)**2),'\n')
    else:
        V = solver.solve(problem, x=V0)
        
    check_orthogonal(V)
    
    results = {'V': V, 'S': V.dot(X_w), 'A': Whiten_mat_inv.dot(V.T)}
    
    return results

def V_manifold_optim(X_w,
                     num_sources,
                     verbose, 
                     constraints, 
                     Whiten_mat,
                     Whiten_mat_inv,
                     result_traceback, 
                     grad_eps=1e-1, 
                     k=50, 
                     reg=False, 
                     **kwargs):
    
    A_entropy_constraint = 'less-entropy' in constraints
    l = num_sources
    d, n = X_w.shape

    try:
        last_results = result_traceback[-1]
        V0 = last_results['V']
    except:
        try:
            V0 = Whiten_mat.dot(last_results['A']).T
        except:
            print('Failed recovering V from last computation, initializing with identity instead...')
            V0 = np.eye(d)
            assert l==d, "Case num_sources!=d not yet implemented"
        
    # (1) Instantiate a manifold
    if d==l:
        if np.linalg.det(V0)<0:
            print('Negative determinant for V0:', np.linalg.det(V0)) if verbose else None
            V0[:,0] *= 1
        manifold = Rotations(d, k=1)
    else:
        manifold = Stiefel(d, l)
    ############################# Question : should we use only rotations to go faster ?!
    
    # (2) Define the cost function
    def cost(V):
        return mutual_information(V.dot(X_w), k=k)
    
    def cost_reg(V):
        b = 1e-2
        return cost(V) + 1/(np.sum(approx_fgrad(V, cost, eps=grad_eps, centered=True))**2 + b)
    
    def cost_euclidian_grad(V):
        if True:
            return approx_fgrad(V, cost, eps=grad_eps, centered=True) 
        else:
            return np.reshape(nd.Gradient(np.vectorize(lambda V: cost(np.reshape(V, (d,l))), 
                                                       signature='(m)->()'))(np.array(V.flat)), (d,l))
        
    def cost_euclidian_hess(V, p_mat):
        d, l = V.shape
        fun = np.vectorize(lambda V: cost(np.reshape(V, (d,l))), signature='(m)->()')
        hess_mat = nd.Hessian(fun)(np.array(V.flat))
        return np.reshape(hess_mat.dot(np.array(p_mat.flat)), (d,l))
    
    def cost_reg_euclidian_grad(V):
        return approx_fgrad(V, cost_reg, eps=grad_eps, centered=True) 
        
    if reg:
        problem = Problem(manifold=manifold, cost=cost_reg, egrad=cost_reg_euclidian_grad)
    else:
        problem = Problem(manifold=manifold,
                          cost=cost,
                          egrad=cost_euclidian_grad,
                          ehess=cost_euclidian_hess,
                          verbosity=2*int(verbose))

    # (3) Instantiate a Pymanopt solver
    # ConjugateGradient is way better than SteepestDescent
    solver = ConjugateGradient(logverbosity=2*int(verbose))

    # let Pymanopt do the rest
    if verbose:
        V, loginfo = solver.solve(problem, x=V0)
        print('I(V.dot(X_w)=', cost(V))
        print('||V-V_0||^2=', np.sum((V-V0)**2),'\n')
    else:
        V = solver.solve(problem, x=V0)
        
    check_orthogonal(V)
    
    results = {'V': V, 'S': V.dot(X_w), 'A': Whiten_mat_inv.dot(V.T)}
    
    return results