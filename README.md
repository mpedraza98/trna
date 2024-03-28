# tRNA gene set evolution

This repo contain files and plots used in the tRNA gene set evolution project

## Files

- **2_state_3_params.py** :
  
  - Solves the equations for the two state system with the parameters $\alpha = \frac{r_M}{r_F}$, $\mu_{M\rightarrow F}$, $\mu_{F\rightarrow M}$. The adimensional time step is given by $\tau = r_F\times t$

  - Plot the final daily population fractions throught the experiment run (including sliders to modify parameters values)
  - Plot the phase space trajectories (with sliders to modify initial conditions)
- **interactive_plot.py** : the same as 2_state_3_params but with sliders for each parameters

  - Solves the equations for the two state system given the parameters $r_M, r_F$, $\mu_{M\rightarrow F}$, $\mu_{F\rightarrow M}$
  - Plot the final daily population fractions throught the experiment run (including sliders to modify parameters values)
  - Plot the phase space trajectories (with sliders to modify initial conditions)

- **three_species_system** :
  - Solves the original three states system given the parameters $r_F, r_D, r_S, \mu_{F\rightarrow D}, \mu_{F\rightarrow S}, \mu_{D\rightarrow F}, \mu_{S\rightarrow F}$
  
  .npy saves numpy arrays (results from code)
  (.h5 files, in the old files folder : saves some intermediate steps for the full inference)
  EvolutionExperiment: an attempt at centralizing everything
  
  parameter_space_explore:script that generates data for the proportion fate + time to convergence (parameter space = muMF vs muRM)

  parameter_space_exploration.ipynb: loads the solution from previous scripts (and also below, both definitions of the parameter space), calculate the analytical solution, and makes the plots (  solution_muMF_alpha: for each point of the parameter space alpha (x) muMF (y), gives the final proportions for the 2 state system
  
  pse: parameter space exploration
  gc: growth curves
  
  pse_2st_muMF_vs_alpha : script that generates data for the proportion fate + time to convergence or each point of the parameter space, space being mu vs alpha
  
  
