import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.widgets import Button, Slider
plt.rcParams['text.usetex'] = True

class EvolutionExperimentTwoStates():
    '''
    The evolution experiment responds to the dynamic given by the equation
    dF/dt = (1 - mu_fc) F + a * mu_cf * C
    dC/dt = mu_fc * F + a (1 - mu_cf) * C

    where t is the time i units of founder replication time

    Takes as input:
        name: name of the model
        t: time array
        params: dictionary with 5 parameters, the replication rates(r_f,r_c), transition rates(mu_fc, mu_cf), carrying capacity(K)
        p0: inital point
        p1: initial point for the subsequent runs of the evolution experiment
        a: alpha (ratio of the mutant's replication rate to the founder's replication rate)
    '''
    # For the equations I will take the time to be in minutes, considering that the replication rates found are in minutes
    
    def __init__(self, strain_name, number_days, model_params, dilution_percentage = 1e-3) -> None:
        
        # Parameters of the model
        # The default for the transition rates is the value from
        # Mutations per generation for wild type (https://doi.org/10.1093/gbe/evu284)
        self.r_f = model_params.get('r_f',0)                        # Founder's replication rate
        self.r_c = model_params.get('r_c', 0)                       # Mutant state replication rate
        self.__mu_fc = model_params.get('mu_fc', 4.25e-9)           # Founder -> Mutant transition rate
        self.__mu_cf = model_params.get('mu_cf', 4.25e-3)           # Mutant -> Founder transition rate
        self.K = model_params.get('K', 1e2)                         # Carrying capacity, 10^10 for the experiment
        self.p0 = np.array([0, 0])                                  # Initial population, in scale of 10^8
        self.strain_name = strain_name
        self.number_days = number_days
        self.dilution_percentage = dilution_percentage
        
        # Additional variables not given as input
        self.day = 0
        self.time_interval = np.arange(0, int(24 * 60 * self.r_f))
        self.daily_fraction = np.zeros((self.number_days, 2))
        self.history = np.zeros((self.number_days, self.time_interval.shape[0], 2))
        self.history_fraction = np.zeros((self.number_days, self.time_interval.shape[0], 2))

        # Private variables of the class
        self.__frac = 0
        self.__p1 = 0
        self.__alpha0 = self.r_c / self.r_f
        self.__alpha = self.__alpha0
        
        
        # Temporarily stores the solution of the model
        self.sol = 0
    
    #Private variables from the class
    @property
    def frac(self):
        '''
            Contains the ratios of founder and mutant bacterias for a day of growth
        '''
        temp_frac = self.sol / self.sol.sum(axis = 1)[:, None]
        self.__frac = temp_frac
        return self.__frac
    
    @property
    def alpha(self):
        return self.__alpha
    
    @alpha.setter
    def alpha(self, value):
        if value is not None:
            self.__alpha = value
        else:
            self.__alpha = self.r_c / self.r_f
    
    @property
    def mu_fc(self):
        return self.__mu_fc
    
    @mu_fc.setter
    def mu_fc(self, value):
        self.__mu_fc = value / np.log(2)

    @property
    def mu_cf(self):
        return self.__mu_cf
    
    @mu_cf.setter
    def mu_cf(self, value):
        self.__mu_cf = value / np.log(2)
        

    def model(self, vars, t = None):
        '''
            Input:
                vars: array of shape (2,) contains the values of (F, C)
                t: time array, (added as a requirement of odeint, not neccessary in the model)
            Output:
                Values of dF/dt and dC/dt for the specified parameters
        '''
        #print(f"Model alpha :{self.alpha}")
        temp_F, temp_C = vars   
        M = np.array([(1 - self.mu_fc / np.log(2)) * temp_F + self.mu_cf / np.log(2) * self.alpha * temp_C, 
                      self.mu_fc / np.log(2) * temp_F + self.alpha * (1 - self.mu_cf / np.log(2)) * temp_C])
        return M * (1- (temp_F + temp_C) / self.K)

    def solve(self):
        '''
            Integrator for the system of equations
            Temporarily stores the solution in the sol variable
        '''
        #print(f"Alpha : {self.alpha}")
        sol = odeint(self.model, y0 = self.__p1, t = self.time_interval)
        self.sol = sol
    
    def run_experiment(self):
        '''
            Solves the evolution experiment for the specified number of days
            Stores the daily history of populations F and C in the variable history
            history fraction contains the same information as history but in fraction of total population
            
            At the end of each day, the last recorded value of population is diluted by a specified factor
            and then used as initial value for next day's run

            Stores the final values for the fraction of F and C in daily_frac
        '''
        print("Running the evolution experiment")
        self.__p1 = self.p0.copy()
        for day in np.arange(self.number_days):
            self.solve()
            self.history[day] = self.sol
            self.history_fraction[day] = self.sol / self.sol.sum(axis = 1)[:, None]
            self.__p1 = self.sol[-1] * self.dilution_percentage
            self.daily_fraction[day] = self.sol[-1] / self.sol[-1].sum()
            self.day += 1
    
    def run_experiment_for_inference(self, new_alpha, new_mu_fc, new_mu_cf, x = 0):
        '''
            Runs the experiment for inference of the parameters alpha, mu_fc, mu_cf
        '''
        self.alpha = new_alpha
        self.mu_cf = new_mu_cf
        self.mu_fc = new_mu_fc
        self.__p1 = self.p0.copy()
        for day in np.arange(self.number_days):
            self.solve()
            self.__p1 = self.sol[-1] * self.dilution_percentage
            self.daily_fraction[day] = self.sol[-1] / self.sol[-1].sum()

    ## Plotting routines from this point on
    def plot_sol(self, ax):
        ax.plot(self.time_interval, self.sol, label = ['F', 'C'])
        ax.set_title(self.name)
        #ax.set_ylabel('Population(x10^8)')
        #ax.set_xlabel('Time')
        ax.legend()
        return ax
    
    def plot_frac(self):
        fig, ax = plt.subplots(figsize = (10, 8))
        for i in range(self.number_days):
            ax.plot(self.time_interval, self.history_fraction[i], label = ['F', 'C'])
        ax.set_title(self.strain_name)
        #ax.set_ylabel('Population(x10^8)')
        #ax.set_xlabel('Time')
        #ax.legend()
        return ax

    def bar_plot_frac(self):
        fig, ax = plt.subplots(figsize = (10, 8))
        ax.bar(np.arange(self.number_days), self.daily_fraction[:, 0],  label = 'Large')
        ax.bar(np.arange(self.number_days), self.daily_fraction[:, 1], bottom = self.daily_fraction[:, 0] , label = 'Small')
        ax.set_ylim([0, 1])
        ax.legend();

    def plot_evolution_frac(self, interactive = False):
        timeline = np.arange(self.number_days * 24 * 60)
        days = np.arange(self.number_days) * self.time_interval.shape[0]
        fig, ax = plt.subplots(figsize = (10, 8))
        founder_line = ax.plot(days, self.daily_fraction[:, 0], label = 'Founder')[0]
        mutant_line = ax.plot(days, self.daily_fraction[:, 1], label = 'Mutant')[0]
        founder_daily = ax.plot(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 0], '--')[0]
        mutant_daily = ax.plot(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 1], '--')[0]
        ax.set_title(self.strain_name);
        ax.set_ylabel('Population fraction');
        #ax.set_xticks(days, np.arange(self.number_days))
        ax.set_xlabel('Day');
        ax.legend();

        if interactive:
            fig.subplots_adjust(left = 0.1, bottom = 0.3)

            # Make a horizontal slider to control the frequency.
            ax_alpha = fig.add_axes([0.15, 0.1, 0.65, 0.03])
            alpha_slider = Slider(
                ax = ax_alpha,
                label = r'$r_M$',
                valmin = 0,
                valmax = 2,
                valinit = self.__alpha0,
                #valinit = 1
            )
            ax_mu_fc = fig.add_axes([0.15, 0.15, 0.65, 0.03])
            mu_fc_slider = Slider(
                ax = ax_mu_fc,
                label = r'$\mu_{F\rightarrow M}$',
                valmin = 1e-11,
                valmax = 1e-6,
                valinit = 1e-9,
            )
            # Make a vertically oriented slider to control the amplitude
            ax_mu_cf = fig.add_axes([0.15, 0.20, 0.65, 0.03])
            mu_cf_slider = Slider(
                ax = ax_mu_cf,
                label = r'$\mu_{M\rightarrow F}$',
                valmin = 4.25e-3,
                valmax = 4.25e-2,
                valinit = self.mu_cf,
            )
            # The function to be called anytime a slider's value changes
            def update(val):
                self.mu_fc = mu_fc_slider.val
                self.mu_cf = mu_cf_slider.val
                self.__alpha = alpha_slider.val
                self.run_experiment()
                founder_line.set_ydata(self.daily_fraction[:, 0])
                mutant_line.set_ydata(self.daily_fraction[:, 1])
                founder_daily.set_ydata(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 0])
                mutant_daily.set_ydata(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 1])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # The function to be called upon pressing the reset button
            def reset(event):
                mu_fc_slider.reset()
                mu_cf_slider.reset()
                alpha_slider.reset()
                print("RESET")
                print(self.mu_cf)
                print(self.mu_fc)
                self.run_experiment()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                

            # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
            resetax = fig.add_axes([0.8, 0.05, 0.1, 0.04])
            button = Button(resetax, 'Reset', hovercolor='0.975')

            # register the update function with each slider
            mu_fc_slider.on_changed(update)
            mu_cf_slider.on_changed(update)
            alpha_slider.on_changed(update)
            
            button.on_clicked(reset)
        #plt.show()
            
            return ax, button
        
        return ax
    
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
        #ax.set_yscale('log');
        #ax.set_xscale('log');
        ax.legend();
        fig.colorbar(matplotlib.cm.ScalarMappable(cmap = cmap, norm = matplotlib.colors.Normalize(vmin = 1, vmax = self.number_days)), ax= ax, ticks = np.arange(1, self.number_days+1, self.number_days // 5), label = 'Days')
        #self.vector_field_plot(ax)
        if interactive:
            fig.subplots_adjust(left=0.1, bottom=0.2)

            # Make a horizontal slider to control the frequency.
            ax_f0 = fig.add_axes([0.15, 0.05, 0.65, 0.03])
            f0_slider = Slider(
                ax=ax_f0,
                label=r'$F_0$',
                valmin=0,
                valmax=2,
                valinit=self.p0[0],
            )
            ax_c0 = fig.add_axes([0.15, 0.1, 0.65, 0.03])
            c0_slider = Slider(
                ax=ax_c0,
                label=r'$C_0$',
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
        #plt.show()
        return ax

    def vector_field_plot(self, ax = None):
            interval = np.linspace(0, 30, 40)
            F, C = np.meshgrid(interval, interval)

            DF, DC = self.model([F, C])
            N = np.sqrt(DF**2 + DC **2)
            N[N==0] = 1
            DF /= N
            DC /= N
            if ax is None:
                fig, ax = plt.subplots(figsize=(10,6))
            ax.quiver(F, C, DF, DC, headwidth = 2)
            ax.plot(interval, self.K - interval, 'r')
            ax.set_ylim(ymin = 0.5)
            plt.show()

    def __str__(self) -> str:
        return "strain_name : {} \nparameters (r_f : {}, mu_fc : {}, r_c : {}, mu_cf : {}, K : {}) \n \
                ".format(self.strain_name, self.r_f, self.mu_fc, self.r_c, self.mu_cf, self.K)



