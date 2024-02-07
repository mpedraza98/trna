import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.integrate import odeint
from matplotlib.widgets import Button, Slider
plt.rcParams['text.usetex'] = True

# SET THE WORKING DIRECTORY
CWD = '/Users/miguel/Documents/Internship_CENTURI'
os.chdir(CWD)
## DEFINES WORKING CONSTANTS
#FILENAME = args.filename
FILENAME = 'plate_counts.csv'
SAVE_FN = FILENAME.strip().split('.')[0]
START = 105 # Start of the linear portion of the graph
#ALPH_DICT = {'A':0, 'B':1, 'C':2, 'D':3, 'E':3, 'F':5, 'G':6, 'H':7}
delserCGA_path = os.path.join(os.getcwd(), "data", 'fit_gc_delserCGA.csv')
df_fit_delserCGA = pd.read_csv(delserCGA_path, index_col=0)


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
        self.daily_fraction = np.zeros((self.number_days, 2))
        self.history = np.zeros((self.number_days, EvolutionExperiment.time_interval.shape[0], 2))

        self.__frac = 0
        self.p0 = 0
        self.__p1 = 0
        
        
        # Parameters of the model
        # The default for the transition rates is the value from
        # Mutations per generation for wild type (https://doi.org/10.1093/gbe/evu284)
        self.r_f = model_params.get('r_f',0)
        self.mu_fc = model_params.get('mu_fc', 4.25e-9)
        self.r_c = model_params.get('r_c', 0)
        self.mu_cf = model_params.get('mu_cf', 4.25e-9)
        self.K = model_params['K']

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
    

    def model(self, vars, t):
        # Define the system of equations     
        M = np.array([[self.r_f * (1 - self.mu_fc / np.log(2)), self.mu_cf / np.log(2) * self.r_c], [self.r_f * self.mu_fc / np.log(2), self.r_c * (1 - self.mu_cf / np.log(2))]])
        return M.dot(vars) * (1- vars.sum() / self.K)

    def solve(self):
        # Solve the system
        sol = odeint(self.model, y0 = self.__p1, t = EvolutionExperiment.time_interval)
        self.sol = sol
    
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
        fig, ax = plt.subplots(figsize = (10, 8))
        founder_line = ax.plot(days, self.daily_fraction[:, 0], label = 'Founder')[0]
        mutant_line = ax.plot(days, self.daily_fraction[:, 1], label = 'Mutant')[0]
        ax.set_title(self.name);
        ax.set_ylabel('Population fraction');
        ax.set_xlabel('Day');
        ax.legend();

        if interactive:
            fig.subplots_adjust(left=0.1, bottom=0.3)

            # Make a horizontal slider to control the frequency.
            ax_rf = fig.add_axes([0.15, 0.05, 0.65, 0.03])
            rf_slider = Slider(
                ax=ax_rf,
                label=r'$r_F$',
                valmin=0,
                valmax=2,
                valinit=self.r_f,
            )
            ax_rc = fig.add_axes([0.15, 0.1, 0.65, 0.03])
            rc_slider = Slider(
                ax=ax_rc,
                label=r'$r_M$',
                valmin=0,
                valmax=2,
                valinit=self.r_c,
            )
            ax_mu_fc = fig.add_axes([0.15, 0.15, 0.65, 0.03])
            mu_fc_slider = Slider(
                ax=ax_mu_fc,
                label=r'$\mu_{F\rightarrow M}$',
                valmin=4.25e-9,
                valmax=4.25e-8,
                valinit=self.mu_cf,
            )
            # Make a vertically oriented slider to control the amplitude
            ax_mu_cf = fig.add_axes([0.15, 0.20, 0.65, 0.03])
            mu_cf_slider = Slider(
                ax=ax_mu_cf,
                label=r'$\mu_{M\rightarrow F}$',
                valmin=4.25e-9,
                valmax=4.25e-8,
                valinit=self.mu_cf,
            )
            # The function to be called anytime a slider's value changes
            def update(val):
                self.mu_fc = mu_fc_slider.val
                self.mu_cf = mu_cf_slider.val
                self.r_f = rf_slider.val
                self.r_c = rc_slider.val
                self.run_experiment()
                founder_line.set_ydata(self.daily_fraction[:, 0])
                mutant_line.set_ydata(self.daily_fraction[:, 1])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # The function to be called upon pressing the reset button
            def reset(event):
                mu_fc_slider.reset()
                mu_cf_slider.reset()
                rc_slider.reset()
                rf_slider.reset()
                print("RESET")
                print(self.mu_cf)
                print(self.mu_fc)
                self.run_experiment()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                

            # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
            resetax = fig.add_axes([0.8, 0.01, 0.1, 0.04])
            button = Button(resetax, 'Reset', hovercolor='0.975')

            # register the update function with each slider
            mu_fc_slider.on_changed(update)
            mu_cf_slider.on_changed(update)
            rc_slider.on_changed(update)
            rf_slider.on_changed(update)
            
            button.on_clicked(reset)
        plt.show()
    
    def phase_plot(self, interactive = False):
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots(figsize = (10, 6), dpi = 150)
        
        lines = []
        for i in np.arange(self.number_days):
            temp_coords = self.history[i]
            temp_coords = np.append(temp_coords, np.array([temp_coords[-1] * self.dilution_percentage]), axis = 0)
            lines.append(ax.plot(temp_coords[:,1], temp_coords[:,0], ':', alpha = 0.5, c = cmap(i/self.number_days))[0])
        
        test_c = np.linspace(*ax.get_xlim())
        test_f = self.K - test_c
        ax.plot(test_c, test_f, 'r-', lw = 1, label = r'$F = K- C$')
        
        ax.set_xlabel('C');
        ax.set_ylabel('F');
        ax.legend();
        fig.colorbar(matplotlib.cm.ScalarMappable(cmap = cmap, norm = matplotlib.colors.Normalize(vmin = 1, vmax = self.number_days)), ax= ax, ticks = np.arange(1, self.number_days+1, self.number_days // 5), label = 'Days')
        if interactive:
            fig.subplots_adjust(left=0.1, bottom=0.2)

            # Make a horizontal slider to control the frequency.
            ax_f0 = fig.add_axes([0.15, 0.05, 0.65, 0.03])
            f0_slider = Slider(
                ax=ax_f0,
                label=r'$r_F$',
                valmin=0,
                valmax=2,
                valinit=self.p0[0],
            )
            ax_c0 = fig.add_axes([0.15, 0.1, 0.65, 0.03])
            c0_slider = Slider(
                ax=ax_c0,
                label=r'$r_M$',
                valmin=0,
                valmax=2,
                valinit=self.p0[1],
            )
            # The function to be called anytime a slider's value changes
            def update(val):
                self.p0 = np.array([f0_slider.val, c0_slider.val])
                self.run_experiment()
                for i in range(self.number_days):
                    temp_coords = self.history[i]
                    temp_coords = np.append(temp_coords, np.array([temp_coords[-1] * self.dilution_percentage]), axis = 0)
                    lines[i].set_ydata(temp_coords[:,0])
                    lines[i].set_xdata(temp_coords[:,1])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # The function to be called upon pressing the reset button
            def reset(event):
                f0_slider.reset()
                c0_slider.reset()
                self.run_experiment()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                

            # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
            resetax = fig.add_axes([0.8, 0.01, 0.1, 0.04])
            button = Button(resetax, 'Reset', hovercolor='0.975')

            # register the update function with each slider
            f0_slider.on_changed(update)
            c0_slider.on_changed(update)
            
            button.on_clicked(reset)
        plt.show()


    
    def run_experiment(self):
        print("Running the evolution experiment")
        self.__p1 = self.p0.copy()
        for day in np.arange(self.number_days):
            self.solve()
            self.history[day] = self.sol
            self.__p1 = self.sol[-1] * self.dilution_percentage
            self.daily_fraction[day] = self.sol[-1] / self.sol[-1].sum()
            self.day += 1

    def __str__(self) -> str:
        return "name : {} \nparameters (r_f : {}, mu_fc : {}, r_c : {}, mu_cf : {}, K : {}) \n \
                ".format(self.name, self.r_f, self.mu_fc, self.r_c, self.mu_cf, self.K)


test_mu_cf = np.linspace(4.25e-9, 4.25e-8, 10)
test_mu_fc = np.linspace(4.25e-9, 4.25e-8, 10)
test_p0 = np.array([0.001, 0])
num_days = 10
# When using delserCGA the replication rate of the founder is found using the growth curve fits
# Additionally, the replication rate of the mutant is assumed to be the rate of M2lop obtained with the gc fit
# The number of days in the experiment is 100, for testing let's assume 10
# M2lop replication rate : 0.05447838370459147
# delserCGA replication rate : 0.04060341705556068

test_params = {'r_f' : 0.04060, 'r_c' : 0.05448, 'mu_fc' : 4.25e-9, 'K' : 21.31}
model_experiment = EvolutionExperiment('delserCGA', num_days , test_params)
model_experiment.p0 = test_p0

model_experiment.run_experiment()
#model_experiment.plot_evolution_frac(interactive = True)
model_experiment.phase_plot(interactive= True)