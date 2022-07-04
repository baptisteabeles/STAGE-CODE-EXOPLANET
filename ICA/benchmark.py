import autograd.numpy as np
from utils import mean
from core import CustomICA
import matplotlib.pyplot as plt
from time import time
import json

def get_sharp(n, d=1, k=0.2, theta=1):
    # var(gamma) = k.theta^2
    # var(gamma.unif) = E(gamma^2.unif^2)-E(gamma.unif)
    # var(gamma.unif) = E(gamma^2).E(unif^2)-E(gamma).E(unif) = E(gamma^2).E(1) = E(gamma^2) = var(gamma) + E(gamma)^2
    # var(gamma.unif) = k.theta^2 + (k.theta)^2
    if d>1:
        return np.array(list(map(lambda x: get_sharp(n=n, d=1, k=k, theta=theta), range(d))))
    else:
        var = k*(theta**2) + (k*theta)**2
        return np.random.gamma(k,theta,n)*(2*np.random.randint(2, size=n)-1) / (n*var)**.5
    
def get_ring(n, r=1, sigma=0.1):
    phis = np.random.uniform(low=0, high=2*np.pi, size=n)
    rads = r*(1+np.random.normal(scale=sigma, size=n))
    return np.array([rads*np.cos(phis), rads*np.sin(phis)])

def get_disk(n, r=1, k=0.5, theta=1):
    phis = np.random.uniform(low=0, high=2*np.pi, size=n)
    rads = r*np.random.gamma(k, theta, n)
    return np.array([rads*np.cos(phis), rads*np.sin(phis)])

def prepare_sources(sources):
    for s in sources:
        s -= np.average(s)
        s /= (s.dot(s.T))**.5
    return np.array(sources)

def get_gaussian_noise(n, d, cov):
    if len(cov.shape)==1:
        cov = np.diag(cov)
    return np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=n).T/(n**.5)

def match_with_sources(result, solution, verbose=False):
    
    if verbose:
        try:
            print('\n-- Matching results of method {} --\n'.format(result['method']))
        except:
            pass
    
    s = np.copy(solution['S'])
    s_hat = np.copy(result['S'])
    a = np.copy(solution['A'])
    a_hat = np.copy(result['A'])
    
    l, n = s.shape
    d, _ = a.shape
    
    # substract off the mean for each dimension
    s, s_av = mean(s)
    s_hat, s_hat_av = mean(s_hat)
    
    # normalize the sources estimated variance
    s_hat_power = np.average(s_hat**2, axis=1)**0.5
    s_hat_power_n = np.stack(n*(s_hat_power,), axis=-1)
    s_hat *= 1/s_hat_power_n
    a_hat *= np.stack(d*(s_hat_power,), axis=0)  # warning: we multiply columns of A (so stack over axis 0)
    
    s_power = np.average(s**2, axis=1)**0.5
    s_power_n = np.stack(n*(s_power,), axis=-1)
    s *= 1/s_power_n
    a *= np.stack(d*(s_power,), axis=0)          # warning: we multiply columns of A (so stack over axis 0)
    
    covs = s.dot(s_hat.T)/n
    abs_cov = abs(covs)
    print('Covariance matrix between real sources and estimated sources\n', covs) if verbose else None
    
    s_hat_ordered = np.zeros((l,n))
    a_hat_ordered = np.zeros((d,l))
    
    for i in range(l):
        ind = np.argmax(abs_cov[i,:])
        print("Source ",i," labelled as ", ind, " cov:", covs[i,ind]) if verbose else None
        s_hat_ordered[i,:] = s_hat[ind,:]*np.sign(covs[i,ind])
        a_hat_ordered[:,i] = a_hat[:,ind]*np.sign(covs[i,ind])
        if np.argmax(abs_cov[:,ind])!=i:
            print("But estimated source ", ind, " is closer to source ", np.argmax(abs_cov[:,ind]))
            

    s_normalized_error = (np.average((s-s_hat_ordered)**2)/np.average(s**2))**.5
    a_normalized_error = (np.average((a-a_hat_ordered)**2)/np.average(a**2))**.5
    s_hat_matched = s_hat_ordered*s_power_n + np.stack(n*(s_av,), axis=-1)
    a_hat_matched = a_hat_ordered/np.stack(d*(s_power,), axis=-1)
    
    if verbose:
        print('\nS normalized error:', 100*s_normalized_error, '%')
        print('A normalized error:', 100*a_normalized_error, '%\n')
    
    return a_hat_matched, s_hat_matched, a_normalized_error, s_normalized_error

def statistical_test(d, l, n, noise_ratio, A, steps, schedule, schedule_kwargs, constraints):
    nb_ops = len(schedule)
    A_errors = np.zeros((nb_ops, steps))
    S_errors = np.zeros((nb_ops, steps))
    
    for step in range(steps):
        print('step ', step)
        
        # generate sources and measures
        S = get_sharp(n=n, d=d, k=0.2, theta=1)  # k and theta are Gamma law parameters
        X = A.dot(S)
        if noise_ratio>0:
            noise = get_gaussian_noise(n=n, d=d, cov=noise_ratio*np.ones(d))
            X += noise

        # apply ICA
        customICA = CustomICA(X,
                              num_sources=l,
                              schedule=schedule, 
                              schedule_kwargs=schedule_kwargs, 
                              constraints=constraints,
                              verbose=False)
        
        # assess performance
        for ind, result in enumerate(customICA.result_traceback):    
            _, _, a_e, s_e = match_with_sources(result=result, solution={'A':A, 'S':S}, verbose=False)
            A_errors[ind, step] = a_e
            S_errors[ind, step] = s_e
    
    return A_errors, S_errors

def hist(d, l, n, noise_ratio, A, steps, schedule, schedule_kwargs, constraints, A_errors, S_errors, save=False):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
    colors = ['green', 'orange', 'red', 'blue']
    alphas = 0.1*np.array([1,1,1,1])
    names = ['ManOptICA', 'FOBI', 'JadeR', 'skFastICA']
    factor = 40
    for ind, method in enumerate(schedule):
        hist, bins = np.histogram(S_errors[ind,:], bins=int(steps/factor), density=True)
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        ax1.hist(S_errors[ind,:], bins=logbins, color=colors[ind], alpha=1, label=names[ind],
                  stacked=True, density=True, histtype='step')
        ax1.hist(S_errors[ind,:], bins=logbins, color=colors[ind], alpha=alphas[ind],
                  stacked=True, density=True)
        hist, bins = np.histogram(A_errors[ind,:], bins=int(steps/factor), density=True)
        
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        ax2.hist(A_errors[ind,:], bins=logbins, color=colors[ind], alpha=1, label=names[ind],
                  stacked=True, histtype='step', density=True)
        ax2.hist(A_errors[ind,:], bins=logbins, color=colors[ind], alpha=alphas[ind],
                  stacked=True, density=True)

        if schedule_kwargs is not None and schedule_kwargs[ind] != dict():
            textstr = '\n'.join([names[ind]+' params:']+list(map(lambda key_val: key_val[0].replace('_', ' ')+'='+str(key_val[1]), schedule_kwargs[ind].items())))
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)

            # place a text box in upper left in axes coords
            fig.text(0.92, 0.75-(ind+1)*0.1, textstr, fontsize=14,
                     verticalalignment='top', bbox=props)

    ax1.set_xscale('log')
    ax1.set_xlabel(xlabel='error $\|S-\hat{S}\|_{\mathcal{M}_{d,n} (R)} /\|S\|_{\mathcal{M}_{d,n} (R)}$',
               x=0.5, y=0.8, fontsize=20)
    ax1.set_ylabel("error frequency $(\%)$", fontsize=20)
    #yticks = ax1.get_yticks()
    #ax1.set_yticklabels(100*yticks*factor/steps)
    ax1.set_title("Error histogram for sources $S$", fontsize=25, pad=20)
    ax1.grid(True)
    
    ax2.set_xscale('log')
    ax2.set_xlabel(xlabel='error $\|A-\hat{A}\|_{\mathcal{M}_{d,l} (R)} /\|A\|_{\mathcal{M}_{d,l} (R)}$',
               x=0.5, y=0.8, fontsize=20)
    ax2.set_ylabel("error frequency $(\%)$", fontsize=20)
    #yticks = ax2.get_yticks()
    #ax2.set_yticklabels(100*yticks*factor/steps)
    ax2.set_title("Error histogram for matrix $A$", fontsize=25, pad=20)
    ax2.grid(True)
    ax2.legend(loc=(1.05,0.8), ncol=1, fancybox=True, fontsize=15)
    
    textstr = '\n'.join((
        r'$n=%.f$' % (n, ),
        r'$d=%.f$' % (d, ),
        r'$l=%.f$' % (l, ),
        r'$\epsilon=%.2f $' % (100*noise_ratio, ) + r'$\%$',
        r'$A=$'+'\n'+str(A),
        #r'$\mathrm{constraints}=$'+'\n' + str(constraints),
        r'$\mathrm{iterations}=%.f$' % (steps, ),)
        )

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    fig.text(0.92, 0.45, textstr, fontsize=14,
             verticalalignment='top', bbox=props)
    
    if save:
        t = int(time())
        name = 'results/histogram_d'+str(d)+'_l'+str(l)+'_n'+str(n)+'_epsilon'+str(100*noise_ratio)+'_t'+str(t)
        params = {'d': d, 'l': l, 'n': n, 'epsilon': noise_ratio, 'steps': steps, 'time': t, 'A': A.tolist(), 
                  'schedule': list(map(lambda method: method.__name__, schedule)), 
                  'schedule_kwargs': schedule_kwargs, 'constraints': constraints, 
                  'A_errors': A_errors.tolist(), 'S_errors': S_errors.tolist()}
        with open(name+'_params.txt', 'w') as outfile:
            json.dump(params, outfile)
        plt.savefig(name+'_fig.jpg', dpi=200, bbox_inches='tight')
        
    return fig