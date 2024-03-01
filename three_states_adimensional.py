import os
import pandas as pd
import matplotlib.pyplot as plt
import mpltern
import numpy as np
import pymc as pm
from scipy.integrate import odeint
from matplotlib.widgets import Button, Slider, TextBox
print(f"Running on PyMC v{pm.__version__}")
plt.rcParams['text.usetex'] = True

# Initialize random number generator
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
#az.style.use("arviz-darkgrid")

# SET THE WORKING DIRECTORY
CWD = '/Users/miguel/Documents/Internship_CENTURI'
os.chdir(CWD)
## DEFINES WORKING CONSTANTS
#FILENAME = args.filename
FILENAME = 'plate_counts.csv'
SAVE_FN = FILENAME.strip().split('.')[0]


class EvolutionExperiment():
    '''
    Takes as input:
        name: name of the model
        t: time array
        params: dictionary with 5 parameters, the replication rates(r_f,r_c), transition rates(mu_fc, mu_cf), carrying capacity(K)
        p0: inital point
        p1: initial point for the subsequent runs of the evolution experiment
    '''
    # For the equations I will take the time to be in minutes, considering that the replication rates found are in minutes
    # Since each experiment lasts a day, I will asume the time interval to be 24*60 long
    time_interval = np.arange(0, 24*60)
    
    def __init__(self, name, number_days, model_params, dilution_percentage = 1e-3) -> None:
        
        self.name = name
        self.number_days = number_days
        self.dilution_percentage = dilution_percentage
        self.day = 0
        self.daily_fraction = np.zeros((self.number_days, 3))
        self.history = np.zeros((self.number_days, EvolutionExperiment.time_interval.shape[0], 3))

        self.__frac = 0
        self.p0 = 0
        self.__p1 = 0
        
        
        # Parameters of the model
        # The default for the transition rates is the value from
        # Mutations per generation for wild type (https://doi.org/10.1093/gbe/evu284)
        self.r_f = model_params.get('r_f', 0.04060)                         # Founder's replication rate
        self.r_d = model_params.get('r_d', 0.05448)                         # Duplication's replication rate
        self.r_s = model_params.get('r_s', 0.01)                            # SNP mutation replication rate
        self.__mu_fd = model_params.get('mu_fd', 4.25e-9) / np.log(2)       # Transition rate F->D
        self.__mu_fs = model_params.get('mu_fs', 4.25e-9) / np.log(2)       # Transition rate F->S
        self.__mu_df = model_params.get('mu_df', 4.25e-9) / np.log(2)       # Transition rate D->F
        self.__mu_sf = model_params.get('mu_sf', 4.25e-9) / np.log(2)       # Transition rate S->F
        self.K = model_params.get('K', 1e2)                                 # Carrying capacity, 10^10 for the experiment

        # Solution of the model
        self.sol = 0
    
    #Private variables from the class
    @property
    def frac(self):
        temp_frac = self.sol / self.sol.sum(axis = 1)[:, None]
        self.__frac = temp_frac
        return self.__frac
    @property
    def p1(self):
        return self.__p1
    
    @p1.setter
    def p1(self, value):
        self.__p1 = value
    
    @property
    def mu_fd(self):
        return self.__mu_fd
    
    @mu_fd.setter
    def mu_fd(self, value):
        self.__mu_fd = value / np.log(2)
    
    @property
    def mu_fs(self):
        return self.__mu_fs
    
    @mu_fs.setter
    def mu_fs(self, value):
        self.__mu_fs = value / np.log(2)
    
    @property
    def mu_df(self):
        return self.__mu_df
    
    @mu_df.setter
    def mu_df(self, value):
        self.__mu_df = value / np.log(2)
    
    @property
    def mu_sf(self):
        return self.__mu_sf
    
    @mu_sf.setter
    def mu_sf(self, value):
        self.__mu_sf = value / np.log(2)

    def model(self, vars, t):
        #Unpack the variables
        F, D, S = vars
        # Define the system of equations     
        M = np.array([self.r_f * (1 - self.mu_fd + self.mu_fs) * F + self.mu_df * self.r_d * D +  self.mu_sf * self.r_s * S,
                      self.r_f * self.mu_fd * F + self.r_d * (1 - self.mu_df) * D,
                      self.r_f * self.mu_fs * F + self.r_s * (1 - self.mu_sf) * D])
        return M * (1- (F + D + S) / self.K)
     
    def solve(self):
        # Solve the system
        sol = odeint(self.model, y0 = self.__p1, t = EvolutionExperiment.time_interval)
        self.sol = sol
    
    def run_experiment(self):
        #print("Running the evolution experiment")
        self.__p1 = self.p0.copy()
        for day in np.arange(self.number_days):
            self.solve()
            self.history[day] = self.sol
            self.__p1 = self.sol[-1] * self.dilution_percentage
            self.daily_fraction[day] = self.sol[-1] / self.sol[-1].sum()
            self.day += 1

    def plot_sol(self, ax):
        ax.plot(EvolutionExperiment.time_interval, self.sol, label = ['F', 'C'])
        ax.set_title(self.name)
        #ax.set_ylabel('Population(x10^8)')
        #ax.set_xlabel('Time')
        ax.legend()
        return ax
    
    def plot_frac(self, ax):
        temp_frac = self.sol / self.sol.sum(axis = 1)[:, None]
        ax.plot(self.time, temp_frac, label = ['F', 'C'])
        ax.set_title(self.name)
        #ax.set_ylabel('Population(x10^8)')
        #ax.set_xlabel('Time')
        ax.legend()
        return ax

    def plot_evolution_frac(self, interactive = False):
        days = np.arange(self.number_days)
        large_frac = self.daily_fraction[:, 1] + self.daily_fraction[:, 2]
        fig, ax = plt.subplots(figsize = (11, 6))

        founder_line = ax.plot(days, self.daily_fraction[:, 0], label = 'Founder')[0]
        large_fraction_line = ax.plot(days, large_frac, label = 'Duplication + SNP')[0]
        duplication_line = ax.plot(days, self.daily_fraction[:, 1], '--' ,label = 'Duplication')[0]
        snp_line = ax.plot(days, self.daily_fraction[:, 2], '--', label = 'SNP')[0]
        ax.set_title(self.name);
        ax.set_ylabel('Population fraction');
        ax.set_xlabel('Day');
        #fig.legend();

        if interactive:
            fig.subplots_adjust(left = 0.1, bottom = 0.3)

            ax_mu_fd = fig.add_axes([0.15, 0.17, 0.15, 0.03])
            mu_fd_text = TextBox(
                ax = ax_mu_fd,
                label = r'$\mu_{F\rightarrow D}$',
                initial = "{:.4e}".format(self.mu_fd)               
            )
            ax_mu_df = fig.add_axes([0.35, 0.17, 0.15, 0.03])
            mu_df_text = TextBox(
                ax = ax_mu_df,
                label = r'$\mu_{D\rightarrow F}$',
                initial = "{:.4e}".format(self.mu_df)               
            )
            
            ax_mu_fs = fig.add_axes([0.55, 0.17, 0.15, 0.03])
            mu_fs_text = TextBox(
                ax = ax_mu_fs,
                label = r'$\mu_{F\rightarrow S}$',
                initial = "{:.4e}".format(self.mu_fs)               
            )
            ax_mu_sf = fig.add_axes([0.75, 0.17, 0.15, 0.03])
            mu_sf_text = TextBox(
                ax = ax_mu_sf,
                label = r'$\mu_{S\rightarrow F}$',
                initial = "{:.4e}".format(self.mu_sf)               
            )

            # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
            resetax = fig.add_axes([0.8, 0.05, 0.1, 0.04])
            button = Button(resetax, 'Reset', hovercolor='0.975')


            # The function to be called anytime a slider's value changes
            def update(val):
                self.mu_fd = float(mu_fd_text.text)
                self.mu_df = float(mu_df_text.text)
                self.mu_fs = float(mu_fs_text.text)
                self.mu_sf = float(mu_sf_text.text)
                self.run_experiment()
                
                # Update the plot
                founder_line.set_ydata(self.daily_fraction[:, 0])
                duplication_line.set_ydata(self.daily_fraction[:, 1])
                snp_line.set_ydata(self.daily_fraction[:, 2])
                large_fraction_line.set_ydata(self.daily_fraction[:, 1] + self.daily_fraction[:, 2])
                #fig.canvas.draw_idle()
                #fig.canvas.flush_events()
            
            # The function to be called upon pressing the reset button
            def reset(event):
                mu_fd_text.reset()
                mu_df_text.reset()
                mu_fs_text.reset()
                mu_sf_text.reset()
                print("RESET")
                self.run_experiment()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                


            # register the update function with each slider
            mu_fd_text.on_submit(update)
            mu_df_text.on_submit(update)
            mu_fs_text.on_submit(update)
            mu_sf_text.on_submit(update)
            
            button.on_clicked(reset)
            
            return fig, ax
        
        return fig, ax

    def phase_plot(self):
        cmap = plt.get_cmap('viridis')
        ax = plt.subplot(projection = "ternary")
        '''
        lines = []
        for i in np.arange(self.number_days):
            temp_coords = self.history[i]
            temp_coords = np.append(temp_coords, np.array([temp_coords[-1] * self.dilution_percentage]), axis = 0)
            lines.append(ax.plot(temp_coords[:,1], temp_coords[:,0], ':', alpha = 0.5, c = cmap(i/self.number_days))[0])
        '''
        #test_c = np.linspace(*ax.get_xlim())
        #test_f = self.K - test_c
        print(self.history[0,:,0].shape)
        ax.plot(self.history[0,:,0] / self.history[0].sum(axis = 1)[:, None], self.history[0,:,1]/ self.history[0].sum(axis = 1)[:, None],self.history[0,:,2]/ self.history[0].sum(axis = 1)[:, None])
        
        #ax.set_xlabel('C');
        #ax.set_ylabel('F');
        #ax.set_yscale('log');
        #ax.set_xscale('log');
        #ax.legend();
        plt.show()
#        fig.colorbar(matplotlib.cm.ScalarMappable(cmap = cmap, norm = matplotlib.colors.Normalize(vmin = 1, vmax = self.number_days)), ax= ax, ticks = np.arange(1, self.number_days+1, self.number_days // 5), label = 'Days')
        #self.vector_field_plot(ax)
    

    def __str__(self) -> str:
        return "name : {} \nparameters (r_f : {}, mu_fc : {}, r_c : {}, mu_cf : {}, K : {}) \n \
                ".format(self.name, self.r_f, self.mu_fc, self.r_c, self.mu_cf, self.K)



test_p0 = np.array([0.001, 0, 0])
num_days = 100
# When using delserCGA the replication rate of the founder is found using the growth curve fits
# Additionally, the replication rate of the mutant is assumed to be the rate of M2lop obtained with the gc fit
# The number of days in the experiment is 100, for testing let's assume 10
# M2lop replication rate : 0.05447838370459147
# delserCGA replication rate : 0.04060341705556068
# Mutations per generation for wild type (https://doi.org/10.1093/gbe/evu284)
# 4.25e-9
# Using the same value for all as a proxy

test_params = {'r_f' : 0.04060, 'r_d' : 0.05448, 'r_s' : 0.035, 'mu_fd' : 4.25e-9, 'mu_fs' : 4.25e-3, 'mu_df' : 4.25e-9, 'mu_sf' : 4.25e-9, 'K' : 1e2}
model_experiment = EvolutionExperiment('delserCGA', num_days , test_params, dilution_percentage= 1e-2)
model_experiment.p0 = test_p0
model_experiment.p1 = test_p0

model_experiment.run_experiment()
#model_experiment.phase_plot()

fig, ax = model_experiment.plot_evolution_frac(interactive = False)

# Comparison with the measurements
df = pd.read_csv('/Users/miguel/Documents/Internship_CENTURI/data/plate_counts.csv')
df = df.sort_values(['founder', 'replicate']).reset_index(drop=True)
temp_df = df[df.founder=='delserCGA']

for i in temp_df.replicate.unique()[:1]:
    temp_df = temp_df[temp_df.replicate == i]
    ax.plot(temp_df.day.values , temp_df.frac_large, 'x', label = f'Replicate {i} large')
    ax.plot(temp_df.day.values , temp_df.frac_small, 'x', label = f'Replicate {i} small')
fig.legend(loc=7)
#fig.tight_layout()
plt.show()


