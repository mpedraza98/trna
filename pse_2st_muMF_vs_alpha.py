import os
from numbalsoda import lsoda_sig, lsoda, dop853
from numba import njit, cfunc
import numpy as np
from scipy.integrate import simpson
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from argparse import ArgumentParser

'''
parser = ArgumentParser()
parser.add_argument("-mu", "--muFM", dest="muFM",
                    help="value for mutation rate", type = float)
args = parser.parse_args()
print(args.muFM)
'''
# Setting up working directory
CWD = '/Users/miguel/Documents/Internship_CENTURI'
os.chdir(CWD)
clear = lambda: os.system('clear')

def verify_convergence(x1 , x2, epsilon = 1e-4):
    area1 = np.array([simpson(x1[:, 0]), simpson(x1[:, 1])])
    area2 = np.array([simpson(x2[:, 0]), simpson(x2[:, 1])])
    norm = np.linalg.norm(area1 - area2)

    return norm < epsilon  


if __name__ == '__main__':
    ### CREATES THE 2D GRID FROM WHERE POINTS WILL BE SAMPLED TO SOLVE THE ODE SYSTEM

    # Dividing the intervals into 10^3 evenly spaced intervals
    alpha = np.arange(1, 2, 1e-3)
    muMF = np.arange(1e-3, 2e-1, (2e-1 - 1e-3) / 1e3)

    ## Take the midpoint of the cell as the sample point 
    alpha_sample = np.column_stack((alpha[:-1], alpha[1:])).mean(axis = 1)
    muMF_sample = np.column_stack((muMF[:-1], muMF[1:])).mean(axis = 1)

    ### DEFINES THE ODE SOLVER VERIFYING THE CONVERGENCE OF THE SOLUTION 

    # Hardcoding K as 100 since it never changes
    @cfunc(lsoda_sig)
    def rhs(t, u, du, p):
        du[0] = ((1 - p[0]) * u[0] + p[1] * p[2] * u[1])*(1 - (u[0] + u[1]) / 100)
        du[1] = (p[0] * u[0] + p[2] * (1 - p[1]) * u[1])*(1 - (u[0] + u[1]) / 100)

    funcptr = rhs.address # address to ODE function
    # Initial parameters
    updated_params = {'r_f' : 0.04060, 'r_c' : 0.05448, 'mu_fc' : 0.0039e-06, 'mu_cf' : 0.05571133365205144, 'K' : 100}
    t_eval = np.arange(0., 24 * 60 * updated_params['r_f']) # times to evaluate solution

    # Store the solutions for each cell in the grid
    # 0: time to convergence in days, 1: final population of founder, 2: final population mutant
    solution = np.zeros((alpha_sample.shape[0], muMF_sample.shape[0], 3))
    errors = []
    full_solutions = []
    ## The choice of this value is motivated by the inference done previously
    ## For convenience we divide it by log(2) in the definition
    mu_FM = 3e-7 / np.log(2)
    print(alpha_sample.shape)
    t3 = time.time()
    with Pool(4) as pool:
        for i, alpha in enumerate(alpha_sample):
            print(f"Solving for alpha : {alpha}")
            for j, mu_MF in enumerate(muMF_sample):
                # Parameters to be modified in each run
                u0 = np.array([1.,0.]) # Initial conditions
                data = np.array([mu_FM, mu_MF / np.log(2), alpha]) # data you want to pass to rhs (data == p in the rhs).

                temp_data = []
                # True if there is convergence
                converged = False
                # Count how many consecutive convergences, if 3 then stop 
                conv = 0
                # Enforces the max number of iterations
                day = 0
                max_iter = 1000

                while day < max_iter and conv <= 3:
                    try:
                        usol, success = lsoda(funcptr, u0, t_eval, data = data)
                        temp_data.append(usol)
                        u0 = usol[-1] * 1e-2
                        day += 1
                        if day >= 2:
                            converged = verify_convergence(temp_data[day-1], temp_data[day-2], epsilon = 1e-5)
                            if converged:
                                conv += 1
                            else:
                                conv = 0
                        # Saves the final value of the population if there is convergence for three straight days
                        # or if it reached the maximum number of days
                            if conv == 3 or day == max_iter:
                                solution[i, j, 0] = day if day < max_iter else max_iter+1
                                solution[i, j, 1] = usol[-1, 0]
                                solution[i, j, 2] = usol[-1, 1]
                        
                    except:
                        print(f"Error while solving the system for mu_MF: {mu_MF}, mu_FM: {mu_FM} in day {day}")
                        errors.append([mu_MF, mu_FM, day])
                        break
            
            with open(f'solution_muMF_alpha.npy', 'wb') as file:
                np.save(file, solution);
            with open(f'errors_muMF_alpha.txt', 'a') as file:
                for line in errors:
                    file.write(f'{line}\n')
            clear()
            print(f"finished alpha: {alpha}, row: {i}")        
    
    t4 = time.time()
    print(f"experiment time : {t4 - t3}")