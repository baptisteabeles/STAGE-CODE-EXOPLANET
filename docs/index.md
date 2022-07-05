# DOCUMENTACION DE BIBLIOTECA

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

## Login

```python

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
```

* Clona el siguiente repositorio : [Repositorio de Biblioteca](https://github.com/baptisteabeles/code-exoplanet-internship)
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

### Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
