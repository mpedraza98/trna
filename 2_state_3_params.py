'''
This file has the work done for the two state model with three parameters

The following system of equations describe the eolution of the population
dF/dt = (1 - mu_fc) F + a * mu_cf * C
dC/dt = mu_fc * F + a (1 - mu_cf) * C

where t is the time in units of founder replication time
The population values are all in scale of 10^8

'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from scipy.integrate import odeint
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
plt.rcParams['text.usetex'] = True

# SET THE WORKING DIRECTORY
CWD = '/Users/miguel/Documents/Internship_CENTURI'
os.chdir(CWD)
## DEFINES WORKING CONSTANTS
DATA = os.path.join(os.getcwd(), "data")
FILENAME = 'plate_counts.csv'
SAVE_FN = FILENAME.strip().split('.')[0]


class EvolutionExperiment():
    '''

    Takes as input:
        name: name of the model
        t: time array
        params: dictionary with 5 parameters, the replication rates(r_f,r_c), transition rates(mu_fc, mu_cf), carrying capacity(K)
        p0: constant inital point
        p1: initial point for each day of the evolution experiment
        a: alpha (ratio of the mutant's replication rate to the founder's replication rate)
    '''
    # For the equations I will take the time to be in minutes, considering that the replication rates found are in minutes
    # Since each experiment lasts a day, I will asume the time interval to be 24*60 long
    
    
    def __init__(self, strain_name, number_days, model_params, dilution_percentage = 1e-3) -> None:
        
        # Parameters of the model
        # The default for the transition rates is the value from
        # Mutations per generation for wild type (https://doi.org/10.1093/gbe/evu284)
        self.r_f = model_params.get('r_f',0)                                    # Founder's replication rate
        self.r_c = model_params.get('r_c', 0)                                   # Mutant state replication rate
        self.__mu_fc = model_params.get('mu_fc', 4.25e-9) / np.log(2)           # Founder -> Mutant transition rate
        self.__mu_cf = model_params.get('mu_cf', 4.25e-3) / np.log(2)           # Mutant -> Founder transition rate
        self.K = model_params.get('K', 1e2)                                     # Carrying capacity, 10^10 for the experiment
        self.p0 = np.array([0, 0])                                              # Initial population, in scale of 10^8
        self.strain_name = strain_name
        self.number_days = number_days
        self.dilution_percentage = dilution_percentage
        
        # Additional variables not given as input
        self.day = 0
        self.time_interval = np.arange(0, int(24 * 60 * self.r_f))                              # Time step in a day in units of r_f
        self.daily_fraction = np.zeros((self.number_days, 2))                                   # Stores the population proportions at the end of every day
        self.history = np.zeros((self.number_days, self.time_interval.shape[0], 2))             # Stores the complete solution to the equations for every day
        self.history_fraction = np.zeros((self.number_days, self.time_interval.shape[0], 2))    # Stores the population's fractions for each time step of each day

        # Private variables of the class
        self.__frac = 0
        self.__p1 = 0
        self.__alpha0 = self.r_c / self.r_f         # Initial value of r_c / r_f, defined as a separate value to preserve it
        self.__alpha = self.__alpha0                # Value of r_c/r_f, used primarily on interactive plots where alpha can be modified
        
        
        # Temporarily stores the solution of the model
        self.sol = 0
    
    #Private variables from the class
    @property
    def frac(self):
        '''
            Calculate the population ratio
            Deprecated
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
    def p1(self):
        return self.__p1
    
    @p1.setter
    def p1(self, value):
        self.__p1 = value
    
    @property
    def mu_fc(self):
        return self.__mu_fc
    
    # To speed up calculations the set value of the replication rates is immediately converted to continuous time
    # by dividing by ln(2)
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
        
        temp_F, temp_C = vars   
        M = np.array([(1 - self.mu_fc ) * temp_F + self.mu_cf * self.alpha * temp_C, 
                      self.mu_fc * temp_F + self.alpha * (1 - self.mu_cf) * temp_C])
        return M * (1- (temp_F + temp_C) / self.K)

    def solve(self):
        '''
            Integrator for the system of equations
            Temporarily stores the solution in the sol variable
        '''
        
        sol = odeint(self.model, y0 = self.p1, t = self.time_interval)
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
        print(self)
        self.__p1 = self.p0.copy()
        for day in np.arange(self.number_days):
            self.solve()
            self.history[day] = self.sol
            self.history_fraction[day] = self.sol / self.sol.sum(axis = 1)[:, None]
            self.__p1 = self.sol[-1] * self.dilution_percentage
            self.daily_fraction[day] = self.sol[-1] / self.sol[-1].sum()
            self.day += 1
   
    ## PLOTTING ROUTINES FROM THIS POINT ON
            
    def plot_sol(self, ax):
        '''
            Plots the solution for a single day, it requires an axis as argument
        '''
        ax.plot(self.time_interval, self.sol, label = ['F', 'M'])
        ax.set_title(self.strain_name)
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

    def plot_evolution_frac(self, interactive = False, history = False):
        '''
            Main plotting function,
            Plots the population's fraction for each day of the experiment
            -> Interactive : Enables sliders that can modify the parameters' values and update the solution in the plot
            -> History: turns on/off the display of the population at each time step

            Return an axes object that can be used to plot other data,
            if Interactive = True returns also a button
        '''

        timeline = np.arange(self.number_days * 24 * 60)
        #
        days = np.arange(self.number_days) * self.time_interval.shape[0]
        
        fig, ax = plt.subplots(figsize = (10, 8))
        founder_line = ax.plot(days, self.daily_fraction[:, 0], label = 'Founder')[0]
        mutant_line = ax.plot(days, self.daily_fraction[:, 1], label = 'Mutant')[0]
        if history:
            founder_daily = ax.plot(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 0], '--')[0]
            mutant_daily = ax.plot(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 1], '--')[0]
        ax.set_title(self.strain_name);
        ax.set_ylabel('Population fraction');
        ax.set_xticks(days[::5], np.arange(self.number_days, step = 5))
        #ax.set_xticks(days, np.arange(self.number_days))
        ax.set_xlabel('Day');
        ax.legend();

        if interactive:
            fig.subplots_adjust(left = 0.1, bottom = 0.3)

            # Make a horizontal slider to control the value of alpha.
            ax_alpha = fig.add_axes([0.15, 0.1, 0.65, 0.03])
            alpha_slider = Slider(
                ax = ax_alpha,
                label = r'$r_M$',
                valmin = 0,
                valmax = 2,
                valinit = self.__alpha0,
                #valinit = 1
            )
            # Make a horizontal slider to control the value of mu_fc.
            ax_mu_fc = fig.add_axes([0.15, 0.15, 0.65, 0.03])
            mu_fc_slider = Slider(
                ax = ax_mu_fc,
                label = r'$\mu_{F\rightarrow M}$',
                valmin = 1e-11,
                valmax = 1e-6,
                valinit = 1e-9,
            )
            # Make a horizontal slider to control the value of mu_cf.
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
                if history:
                    founder_daily.set_ydata(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 0])
                    mutant_daily.set_ydata(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 1])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # The function to be called upon pressing the reset button
            # It's not working properly right now
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
            
            return ax, button
        
        return ax
    
    def plot_absolute_abundance(self, interactive = False, history = False):

        total_time = np.arange(self.number_days * self.time_interval.shape[0])
        days = np.arange(self.number_days) * self.time_interval.shape[0]
        fig, ax = plt.subplots(figsize = (10, 8))
        #founder_line = ax.plot(days, self.daily_fraction[:, 0], label = 'Founder')[0]
        #mutant_line = ax.plot(days, self.daily_fraction[:, 1], label = 'Mutant')[0]
        ax.plot(total_time, self.history.reshape(-1, self.history.shape[-1])[:,0])
        ax.plot(total_time, self.history.reshape(-1, self.history.shape[-1])[:,1])
        #ax.set_yscale('log')
        #ax.plot(self.history.reshape(-1, self.history.shape[-1])[:,1] / self.history.reshape(-1, self.history.shape[-1])[:,0], '-k')
        #ax.plot(self.history.reshape(-1, self.history.shape[-1])[:,0] / self.history.reshape(-1, self.history.shape[-1])[:, 1], '-k')
        ax.axhline(y = self.mu_fc / self.mu_cf, xmin = 0, xmax = total_time[-1])
        #ax.axhline(y = self.mu_cf / self.mu_fc, xmin = 0, xmax = total_time[-1])
        if history:
            founder_daily = ax.plot(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 0], '--')[0]
            mutant_daily = ax.plot(self.history_fraction.reshape((self.time_interval.shape[0] * self.number_days, 2))[:, 1], '--')[0]
        ax.set_title(self.strain_name);
        ax.set_ylabel('Population fraction');
        #ax.set_xticks(days[::5], np.arange(self.number_days, step = 5))
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
                if history:
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

    def phase_plot(self, interactive = False, log_scale = False):
        ''''
            Plot trajectories in the phase space
            The color gradient indicates the experiment day
            Note: This is the only way I found to for visually representing the days, there's probably a better solution I haven't thought of

            -> Interactive : Enables sliders to modify the initial conditions of the experiment
            -> log_scale : changes the plot to log scale, helps visualize the small population changes at the beginning of the experiment

            Returns:
            Axes object
        '''

        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots(figsize = (10, 6), dpi = 150)
        
        lines = []
        for i in np.arange(self.number_days):
            temp_coords = self.history[i]
            temp_coords = np.append(temp_coords, np.array([temp_coords[-1] * self.dilution_percentage]), axis = 0)
            lines.append(ax.plot(temp_coords[:,0], temp_coords[:,1], ':', alpha = 0.5, c = cmap(i/self.number_days))[0])
        
        test_c = np.linspace(*ax.get_xlim())
        test_f = self.K - test_c
        ax.plot(test_c, test_f, 'r-', lw = 1, label = r'$F = K- C$')
        
        ax.set_xlabel('F');
        ax.set_ylabel('M');
        if log_scale:
            ax.set_yscale('log');
            ax.set_xscale('log');
        ax.legend();
        fig.colorbar(matplotlib.cm.ScalarMappable(cmap = cmap, norm = matplotlib.colors.Normalize(vmin = 1, vmax = self.number_days)), ax= ax, ticks = np.arange(1, self.number_days+1, self.number_days // 5), label = 'Days')

        if interactive:
            fig.subplots_adjust(left=0.1, bottom=0.2)

            # Make a horizontal slider to control the initial founder population.
            ax_f0 = fig.add_axes([0.15, 0.05, 0.65, 0.03])
            f0_slider = Slider(
                ax=ax_f0,
                label=r'$F_0$',
                valmin=0,
                valmax=2,
                valinit=self.p0[0],
            )
            # Make a horizontal slider to control the initial mutant population.
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
                self.p1 = np.array([f0_slider.val, c0_slider.val])
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

    def vector_field_plot(self, ax = None, animate = False):
            '''
                Plot the vector field for the system of equations
                
                ax -> set of axis to plot in
                animate -> generates a gif showing the experimental trajectory
            '''

            interval = np.linspace(0, 100, 35)
            F, C = np.meshgrid(interval, interval)

            DF, DC = self.model([F, C])
            N = np.sqrt(DF**2 + DC **2)
            N[N==0] = 1
            DF /= N
            DC /= N
            if ax is None:
                fig, ax = plt.subplots(figsize=(10,6), dpi = 150)
            ax.quiver(F, C, DF, DC, headwidth = 1.7, headlength = 4, alpha = 0.4)
            ax.plot(interval, self.K - interval, "r", label = r'$F = K - M$')
            ax.plot(self.p0[0], self.p0[1], 'rx')
            ax.set_xlabel(r'Founder Population ($\times 10^8$)')
            ax.set_ylabel(r'Mutant Population ($\times 10^8$)')
            #ax.set_ylim(ymin = 0.5)
            
            cmap = plt.get_cmap('viridis')
            #fig, ax = plt.subplots(figsize = (10, 6), dpi = 150)            
            for i in np.arange(self.number_days):
                temp_coords = self.history[i]
                temp_coords = np.append(temp_coords, np.array([temp_coords[-1] * self.dilution_percentage]), axis = 0)
                ax.plot(temp_coords[:,0], temp_coords[:,1], alpha = 0.5, c = cmap(i/self.number_days))[0]

            #ax.set_yscale('log');
            #ax.set_xscale('log');
            ax.legend();
            fig.colorbar(matplotlib.cm.ScalarMappable(cmap = cmap, norm = matplotlib.colors.Normalize(vmin = 1, vmax = self.number_days)), ax= ax, ticks = np.arange(1, self.number_days+1, self.number_days // 5), label = 'Days')

            if animate:
                temp_history = self.history.reshape(-1, 2)
                line, = ax.plot(temp_history[0,0], temp_history[0,1], 'C1', alpha = 0.5)
                dot, = ax.plot(temp_history[0,0], temp_history[0,1], 'ro', alpha = 0.5)

                def animation_function(i):
                    #print(i)
                    ax.set_title(f'Day {i // self.time_interval[-1]}')
                    dot.set_xdata(temp_history[i, 0])
                    dot.set_ydata(temp_history[i, 1])
                    line.set_xdata(temp_history[:i, 0])
                    line.set_ydata(temp_history[:i, 1])
                    return line, 
                animation = FuncAnimation(fig, func = animation_function, frames = np.arange(0,temp_history.shape[0]), interval = 10)
                #animation.save(CWD + '/plots/phase_space_trajectory.gif')

            plt.show()

    def __str__(self) -> str:
        return "strain_name : {} \nparameters (r_f : {}, mu_fc : {}, r_c : {}, mu_cf : {}, K : {}, initial alpha : {}, alpha : {}) \n \
                ".format(self.strain_name, self.r_f, self.mu_fc, self.r_c, self.mu_cf, self.K, self.__alpha0, self.alpha)



if __name__ == '__main__':

    # When using delserCGA the replication rate of the founder is found using the growth curve fits
    # Additionally, the replication rate of the mutant is assumed to be the rate of M2lop obtained with the gc fit
    # The number of days in the experiment is 100, for testing let's assume 10
    # M2lop replication rate : 0.05447838370459147
    # delserCGA replication rate : 0.04060341705556068
    # M2lop carrying capacity from OD: 2204.1792043341115 (x10^8)
    # delserCGA carrying capacity from OD: 21.21143202308204 (x10^8)
    # From the experiments:
    #   K:10^10, adjusting for the scale we're using, effctively 10^2
    #   Bottleneck size (Dilution percentage): 1%

    test_p0 = np.array([1, 0])
    num_days = 101
    test_params = {'r_f' : 0.04060, 'r_c' : 0.05448, 'mu_fc' : 0.0039e-06, 'mu_cf' : 1.26610510e-03, 'K' : 100}
    #test_params = {'r_f' : 0.04060, 'r_c' : 0.05448, 'mu_fc' : 1.36610510e-3, 'mu_cf' : 1.26610510e-01, 'K' : 100}
    model_experiment = EvolutionExperiment('delserCGA', num_days , test_params, dilution_percentage = 0.01)
    model_experiment.p0 = test_p0
    model_experiment.p1 = test_p0
    #model_experiment.alpha = 1


    
    model_experiment.run_experiment()
    #model_experiment.vector_field_plot(animate=False)
    '''
    with open('sol.npy', 'wb') as f:
        np.save(f, model_experiment.history)
    model_experiment.plot_absolute_abundance()
    '''
    #ax = model_experiment.phase_plot(interactive= False)
    #plt.show()
    #ax = model_experiment.plot_evolution_frac(interactive = False)
    ax = model_experiment.plot_evolution_frac(interactive = True)[0]
    #ax.set_xticks([],[])
    #ax.set_ylim(ymin = -0.05, ymax = 1.05)
    #plt.savefig(CWD + '/plots/two_state_big_mu_mf.png', dpi = 300)
    #model_experiment.plot_frac()
    #model_experiment.bar_plot_frac()
    #plt.show()

    
    # In this part of the code I was testing the idea that in the limit of an infinite carrying capacity the experiments should 
    # converge to the same equilibrium state. In principle what the dilution does is just changing the initial conditions of the equations
    # thus, making it easier for the mutant population to grow
    '''
    test_ndays = 100
    model_experiment.K = np.inf
    model_experiment.p1 = test_p0
    print(model_experiment.time_interval.shape)
    model_experiment.time_interval = np.arange(0, test_ndays*int(24 * 60 * model_experiment.r_f))
    print(model_experiment.time_interval.shape)
    model_experiment.solve()
    print(model_experiment.sol.shape)
    fig, ax = plt.subplots(figsize = (10, 6))
    #model_experiment.plot_sol(ax)
    temp_fraction = model_experiment.sol / model_experiment.sol.sum(axis = 1)[:, None]
    print("__________________________")
    print(temp_fraction.shape)
    print(temp_fraction[-10])
    ax.plot(temp_fraction[:,0], label = 'F')
    ax.plot(temp_fraction[:,1], label = 'M')
    #ax.set_xticks(np.arange(test_ndays) * model_experiment.time_interval.shape[0], np.arange(0,test_ndays))
    ax.set_xlabel('Days')
    ax.legend();
    plt.show()
    '''


    # Uncomment this portion of the code to see the vector field plot
    # Creating the animation is a slow process and the gif will take several minutes to render
    '''
    model_experiment.vector_field_plot()

    model_experiment.alpha = 0
    model_experiment.solve()
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(model_experiment.sol[:,0], label = 'Founder')
    ax.plot(model_experiment.sol[:,1], label = 'Mutant')
    plt.legend();
    plt.show()
    
    model_experiment.run_experiment()

    fig, ax = plt.subplots(figsize = (10, 6))
    for i in range(model_experiment.number_days):
        ax.plot(model_experiment.history[i][:,0], '--', label = f"Day{i}")
        ax.plot(model_experiment.history[i][:,1], label = f"Day{i}")

    plt.legend()

    plt.show()
    '''
    

    # Comparison with the measurements
    # Load and plot the experimental data
    # Requires an existing axis object
    df = pd.read_csv('/Users/miguel/Documents/Internship_CENTURI/data/plate_counts.csv')
    df = df.sort_values(['founder', 'replicate']).reset_index(drop=True)
    temp_df = df[df.founder=='delserCGA']
    for i in temp_df.replicate.unique()[:1]:
        temp_df = temp_df[temp_df.replicate == i]
        ax.plot(temp_df.day.values * model_experiment.time_interval.shape[0], temp_df.frac_large, 'x', label = f'Replicate {i} Large')
        ax.plot(temp_df.day.values * model_experiment.time_interval.shape[0], temp_df.frac_small, 'x', label = f'Replicate{i} Small')
    ax.legend()
    plt.show()
