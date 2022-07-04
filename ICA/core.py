import autograd.numpy as np
import matplotlib.pyplot as plt
from utils import plot_2D_distrib, mean, whiten, analyze, mutual_information, SNR
from heuristics import FOBI, noise_optim, V_optim, V_global_optim, V_manifold_optim, sk_FastICA
from itertools import chain, starmap

class PostProcess():
    
    def __init__(self, obj_list, target):
        self.obj_list = list(filter(None, obj_list))
        self.target = target
        self.steps = sum(map(lambda obj: len(obj.schedule), self.obj_list))
        assert abs(np.average(self.target)) < 1e-10, \
               "Warning target not centered: " + str(np.average(self.target))
        if abs(self.target.dot(self.target.T)-1) > 1e-8:
            self.target /= (self.target.dot(self.target.T))**.5
        self.num_sources_list = np.unique(list(map(lambda obj: obj.num_sources, self.obj_list)))
        self.S_results = {l: [] for l in self.num_sources_list}
        self.A_results = {l: [] for l in self.num_sources_list}
        self.signals = {l: [] for l in self.num_sources_list}
        self.grades = {l: [] for l in self.num_sources_list}
        self.entropies = {l: [] for l in self.num_sources_list}
        self.target_corrs = {l: [] for l in self.num_sources_list}
        
        for obj in self.obj_list:
            self.S_results[obj.num_sources] += list(map(lambda result: result['S'], obj.result_traceback))
            self.A_results[obj.num_sources] += list(map(lambda result: result['A'], obj.result_traceback))
            
        list(map(self.gen_signals, self.num_sources_list))
        
            
    def gen_S_trust(self, S):
        l, n = S.shape
        if np.max(abs(np.sum(S, axis=1))) > 1e-10:
            print("Warning S not centered: ", np.max(abs(np.sum(S, axis=1))))
            S, _ = mean(S)
        if np.max(abs(S.dot(S.T)-np.eye(l))) > 1e-8:
            print("Warning S not normalized: ", np.max(abs(S.dot(S.T)-np.eye(l))))
            S /= np.stack(n*(np.diag(S.dot(S.T))**.5,), axis=1)
            print(S.dot(S.T))
            
        covs = S.dot(self.target.T)
        
        if len(covs)>1:
            probs = covs**2/np.sum(covs**2)
            probs_light = probs[probs.nonzero()]
            signal_ind = covs.argmax()
            
            return S[signal_ind,:]*np.sign(covs[signal_ind]), \
                   np.sum(-probs_light*np.log(probs_light)), \
                   covs[signal_ind]
        else:
            return S[0]*np.sign(covs[0]), \
                   1., \
                   covs[0]
        
    def gen_signals(self, num_sources):
        signals, entropies, target_corrs = map(list, zip(*map(self.gen_S_trust, self.S_results[num_sources])))
        self.signals[num_sources] = np.asarray(signals)
        self.entropies[num_sources] = np.asarray(entropies)
        self.target_corrs[num_sources] = np.asarray(target_corrs)                  
        
            
    def post_process(self, amplifier=lambda x: x**0, select_l=None):
        if select_l is None:
            list(map(lambda l: self.post_process(amplifier=amplifier, select_l=l), self.num_sources_list))
            return np.average(np.asarray(list(chain(*self.signals.values()))),
                              weights=np.asarray(list(chain(*self.grades.values()))), 
                              axis=0)
        else:
            self.grades[select_l] = np.array(list(starmap(amplifier, zip(self.target_corrs[select_l],
                                                                         self.entropies[select_l]))))
            return np.average(self.signals[select_l], 
                              weights=self.grades[select_l], 
                              axis=0)
    
    def investigate(self, true_signal, plot_timeline=None, plot_title='true', amplifier=lambda x: x**0):
        if abs(np.sum(true_signal))>1e-10:
            print("Given true signal must be centered to process comparison")
        if abs(true_signal.dot(true_signal.T)-1)>1e-10:
            print("Given true signal must be normalized to process comparison")
    
        plot_timeline = range(len(true_signal)) if plot_timeline is None else plot_timeline
        fig = plt.figure()
        plt.title('Error to '+plot_title+' signal after '+str(self.steps)+' ICA post processed', fontsize=15)
        for l in self.num_sources_list:
            guess_l = self.post_process(amplifier=amplifier, select_l=l)
            plt.scatter(plot_timeline, guess_l, label='Guessed '+str(l)+' signal', marker='x', alpha=0.5)
            print('Guess out of '+str(l)+' sources, epoch='+str(self.steps)+' relative error: ', \
                   100*np.average((guess_l-true_signal)**2)/np.average((true_signal)**2), '%', \
                  ' SNR: ', SNR(true_signal, guess_l))

        guess = self.post_process(amplifier=amplifier)
        plt.scatter(plot_timeline, guess, label='Guessed full signal', marker='x', alpha=0.5)
        print('Guess full, epoch='+str(self.steps)+' relative error: ', \
              100*np.average((guess-true_signal)**2)/np.average((true_signal)**2), '%', \
              ' SNR: ', SNR(true_signal, guess))
        
        l_opt = np.argmax([self.grades[l].max() for l in self.num_sources_list])
        self.best = self.signals[self.num_sources_list[l_opt]][self.grades[self.num_sources_list[l_opt]].argmax()]
        plt.scatter(plot_timeline, self.best, label='Guessed best signal', marker='x', alpha=0.5)
        print('Guess best, epoch=', self.steps, \
              ' cov=', self.target_corrs[self.num_sources_list[l_opt]][self.grades[self.num_sources_list[l_opt]].argmax()], \
              ' entropy=', self.entropies[self.num_sources_list[l_opt]][self.grades[self.num_sources_list[l_opt]].argmax()], \
              ' relative error: ', 100*np.average((self.best-true_signal)**2)/np.average((true_signal)**2), '%', \
              ' SNR: ', SNR(true_signal, self.best))

        avg_err = 100*np.average((self.target-true_signal)**2)/np.average((true_signal)**2)
        plt.scatter(plot_timeline, self.target-true_signal, label='Sensor averaged signal', marker='x', alpha=0.5)
        print('Target relative error: ', avg_err, '%', ' SNR: ', SNR(true_signal, self.target))
        
        plt.grid(True)
        plt.legend()
        
        return fig
    
        

class CustomICA():
    
    def __init__(self, 
                 X_input, 
                 constraints=[],
                 num_sources=None, 
                 noise_fraction=0/100, 
                 noise_add = 0/100,
                 schedule=[sk_FastICA], 
                 schedule_kwargs=None,
                 verbose=True, 
                 plot_loss=lambda X: mutual_information(X, k=50),
                 plot_loss_name='I_{k=50}',
                 make_analyze=True, 
                 skip_exception=False):
        
        self.X = np.copy(X_input)
        d, n = np.shape(self.X)
        self.d = d
        self.n = n
        self.num_sources = num_sources if num_sources is not None else d
        self.noise_fraction = noise_fraction
        self.noise_add = noise_add
        
        assert self.d >= self.num_sources, \
               "Please make sure num_sources<=d"
        assert not (self.num_sources!=d and self.noise_fraction==0/100), \
               "Please allow noise to the model if num_sources is not d"
            
        self.verbose = verbose
        self.constraints = constraints
        self.plot_loss = plot_loss
        self.plot_loss_name = plot_loss_name
        
        self.make_analyze = make_analyze    
        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs if schedule_kwargs is not None else len(self.schedule)*[{}]
        assert len(self.schedule)==len(self.schedule_kwargs), \
               "Please make sure to match methods scheduled and their kwargs."
        
        self.pre_process()
        
        self.result_traceback = []
        self.skip_exception = skip_exception
        
        for method, method_kwargs in zip(self.schedule, self.schedule_kwargs):
            self.apply(method, method_kwargs)
            
        return
            
    def apply(self, method, method_kwargs):
        try:
            step = len(self.result_traceback)
            print('\nStep '+str(step)+': proceeding to {} ...\n'.format(method.__name__)) if self.verbose else None
            new_results = method(**self.__dict__, **method_kwargs)
            new_results['method'] = method.__name__
            new_results['step'] = step
            self.result_traceback.append(new_results)
            analyze(eps=2e-1, **self.__dict__) if self.make_analyze else None
        except Exception as error:
            if not self.skip_exception:
                raise error
            else:
                print(error)
        
      
    def pre_process(self):
        # substract off the mean over time
        self.X, X_av = mean(self.X)
        self.X_av = X_av

        # time signal averaged over all channels:
        self.X_chan_av = np.average(self.X, axis=0)

        if self.verbose:
            ax = plot_2D_distrib(self.X, title='centered', loss=self.plot_loss, loss_name=self.plot_loss_name)
            plt.show()

        # whiten the data: Whiten_mat.dot(X) = X_w  and   X_w.dot(X_w.T)=np.eye(d)
        # X_w data is sorted by decreasing variance
        X_w, Whiten_mat, Whiten_mat_inv, eigs_RY_2 = whiten(self.X, sort=True)  
        self.X_w = X_w
        self.Whiten_mat = Whiten_mat
        self.Whiten_mat_inv = Whiten_mat_inv
        self.eigs_RY_2 = eigs_RY_2

        if self.verbose:
            ax = plot_2D_distrib(self.X_w, title='whitened', loss=self.plot_loss, loss_name=self.plot_loss_name)
            plt.show()


def optim_ICA(X_input, 
              constraints=[],
              num_sources=None, 
              noise_fraction=0/100, 
              schedule=[FOBI, V_manifold_optim], 
              schedule_kwargs=None,
              verbose=True, 
              plot_inf=True,
              A_guess=None):
    
    X = np.copy(X_input)
    d, n = np.shape(X)
    
    if num_sources is None:
        num_sources = d
        
    if schedule_kwargs is None:
        schedule_kwargs = len(schedule)*[{}]
    
    # substract off the mean over time
    X, X_av = mean(X)
    
    # time signal averaged over all channels:
    X_chan_av = np.average(X, axis=0)

    if verbose:
        ax = plot_2D_distrib(X, title='centered', plot_inf=plot_inf)
        plt.show()
    
    # whiten the data: Whiten_mat.dot(X) = X_w  and   X_w.dot(X_w.T)=np.eye(d)
    # X_w data is sorted by decreasing variance
    X_w, Whiten_mat, Whiten_mat_inv, eigs_RY_2 = whiten(X, sort=True)   
    
    if verbose:
        ax = plot_2D_distrib(X_w, title='whitened', plot_inf=plot_inf)
        plt.show()

    ICA_params = {'X': X,        # warning: here X is zero meaned, it is not the input X
                  'X_av': X_av,
                  'X_chan_av': X_chan_av, 
                  'X_w': X_w,
                  'Whiten_mat': Whiten_mat, 
                  'Whiten_mat_inv': Whiten_mat_inv,
                  'eigs_RY_2': eigs_RY_2,
                  'num_sources': num_sources,
                  'constraints': constraints,
                  'noise_fraction': noise_fraction,
                  'verbose': verbose, 
                  'plot_inf': plot_inf, 
                  'A_guess': A_guess, 
                  'schedule': schedule}
    
    result_traceback = []
    for ind, method in enumerate(schedule):
        print('\nProceeding to {} ...\n'.format(method.__name__)) if verbose else None
        new_results = method(ICA_params, result_traceback, **schedule_kwargs[ind])
        new_results['method'] = method.__name__
        new_results['step'] = ind
        analyze(ICA_params, new_results, eps=2e-1)
        result_traceback.append(new_results)

    return result_traceback, ICA_params