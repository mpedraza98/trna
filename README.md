# tRNA gene set evolution

This repo contain files and plots used in the tRNA gene set evolution project

## Files

- **2_state_3_params.py** :
  
  - Solves the equations for the two state system with the parameters $\alpha = \frac{r_M}{r_F}$, $\mu_{M\rightarrow F}$, $\mu_{F\rightarrow M}$. The adimensional time step is given by $\tau = r_F\times t$

  - Plot the final daily population fractions throught the experiment run (including sliders to modify parameters values)
  - Plot the phase space trajectories (with sliders to modify initial conditions)
- **interactive_plot.py** :

  - Solves the equations for the two state system given the parameters $r_M, r_F$, $\mu_{M\rightarrow F}$, $\mu_{F\rightarrow M}$
  - Plot the final daily population fractions throught the experiment run (including sliders to modify parameters values)
  - Plot the phase space trajectories (with sliders to modify initial conditions)

- **three_species_system** :
  - Solves the original three states system given the parameters $r_F, r_D, r_S, \mu_{F\rightarrow D}, \mu_{F\rightarrow S}, \mu_{D\rightarrow F}, \mu_{S\rightarrow F}$
