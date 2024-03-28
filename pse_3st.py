import os
from numbalsoda import lsoda_sig, lsoda, dop853
from numba import njit, cfunc
import numpy as np
from scipy.integrate import simpson
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-b", "--beta", dest="beta",
                    help="value for alpha", type = float)
args = parser.parse_args()
print(args.beta)
# Setting up working directory
CWD = '/Users/miguel/Documents/Internship_CENTURI'
os.chdir(CWD)
clear = lambda: os.system('clear')

def verify_convergence(x1 , x2, epsilon = 1e-4):
    area1 = np.array([simpson(x1[:, 0]), simpson(x1[:, 1]), simpson(x1[:, 2])])
    area2 = np.array([simpson(x2[:, 0]), simpson(x2[:, 1]), simpson(x2[:, 2])])
    norm = np.linalg.norm(area1 - area2)

    return norm < epsilon  


if __name__ == '__main__':
    ### CREATES THE 2D GRID FROM WHERE POINTS WILL BE SAMPLED TO SOLVE THE ODE SYSTEM

    # Dividing the intervals into 10^3 evenly spaced intervals
    muFD = np.arange(1e-8, 1e-6, (1e-6 - 1e-8) / 1e2)
    muFS = np.arange(1e-7, 1e-5, (1e-5 - 1e-7) / 1e2)
    muDF = np.arange(1e-6, 1e-4, (1e-4 - 1e-6) / 1e2)
    muSF = np.arange(1e-5, 1e-3, (1e-3 - 1e-5) / 1e2)

    #X, Y = np.meshgrid(muFM, muMF)

    # Make an array with the boundaries of each cell in the grid, it is neccessary to sample from the uniform distribution
    muFD_interval = np.column_stack((muFD[:-1], muFD[1:]))
    muFS_interval = np.column_stack((muFS[:-1], muFS[1:]))
    muDF_interval = np.column_stack((muDF[:-1], muDF[1:]))
    muSF_interval = np.column_stack((muSF[:-1], muSF[1:]))

    ## Sampling from a uniform distribution over each cell in the grid. The sampled points are different for each cell
    np.random.seed(1234)
    # mu_MF [:,:,0], mu_FM [:,:,1]
    sample_mu = np.zeros((muFD_interval.shape[0], muFS_interval.shape[0], muDF_interval.shape[0], muSF_interval.shape[0],  4))
    for fd0, fd1 in enumerate(muFD_interval):
        for fs0, fs1 in enumerate(muFS_interval):
            for df0, df1 in enumerate(muDF_interval):
                for sf0, sf1 in enumerate(muSF_interval):
                    temp_muFD = np.random.uniform(fd1[0], fd1[1])
                    temp_muFS = np.random.uniform(fs1[0], fs1[1])
                    temp_muDF = np.random.uniform(df1[0], df1[1])
                    temp_muSF = np.random.uniform(sf1[0], sf1[1])
                    sample_mu[fd0, fs0, df0, sf0] = np.array([temp_muFD, temp_muFS, temp_muDF, temp_muSF])

    with open('sample_mu.npy', 'wb') as f:
        np.save(f, sample_mu)

    ### DEFINES THE ODE SOLVER VERIFYING THE CONVERGENCE OF THE SOLUTION 

    

    # Hardcoding K as 100 since it never changes
    @cfunc(lsoda_sig)
    def rhs(t, u, du, p):
        ## u[F, D, S]
        ## alpha = r_D / r_F, beta = r_S / r_F
        ## p[mu_fd, mu_fs, mu_df, mu_sf, alpha, beta]
        du[0] = ((1 - p[0] - p[1]) * u[0] + p[2] * p[4] * u[1] + p[3] * p[5] * u[2])*(1 - (u[0] + u[1] + u[2]) / 100)
        du[1] = (p[0] * u[0] + p[4] * (1 - p[2]) * u[1])*(1 - (u[0] + u[1] + u[2]) / 100)
        du[2] = (p[1] * u[0] + p[5] * (1 - p[3]) * u[2])*(1 - (u[0] + u[1] + u[2]) / 100)

    funcptr = rhs.address # address to ODE function
    # Initial parameters
    updated_params = {'r_f' : 0.04060, 'r_d' : 0.05448, 'r_s' : 0.035, 'mu_fd' : 4.7e-6, 'mu_fs' : 8.1e-5, 'mu_df' : 8.1e-5, 'mu_sf' : 0.00081, 'K' : 1e2}
    t_eval = np.arange(0., 24 * 60 * updated_params['r_f']) # times to evaluate solution
    alpha = updated_params['r_d'] / updated_params['r_f']
    beta = args.beta

    # Store the solutions for each cell in the grid
    # 0: time to convergence in days, 1: final population of founder, 2: final population mutant
    solution = np.zeros((*sample_mu.shape[:-1], 4))
    errors = []
    full_solutions = []

    t3 = time.time()
    with Pool(4) as pool:
        for i in range(sample_mu.shape[0]):
            print(f"Solving for mu_FD = {i}")
            for j in range(sample_mu.shape[1]):
                print(f"Solving for mu_FS = {j}")
                for k in range(sample_mu.shape[2]):
                    print(f"Solving for mu_DF = {k}")
                    for m in range(sample_mu.shape[3]):
                        print(f"Solving for mu_SF = {m}")
                        # Parameters to be modified in each run
                        mu_FD, mu_FS, mu_DF, mu_SF = sample_mu[i, j, k, m]
                        u0 = np.array([1., 0., 0.]) # Initial conditions
                        data = np.array([mu_FD / np.log(2), mu_FS / np.log(2), mu_DF / np.log(2), mu_SF / np.log(2), alpha, beta]) # data you want to pass to rhs (data == p in the rhs).

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
                                    converged = verify_convergence(temp_data[day-1], temp_data[day-2])
                                    if converged:
                                        conv += 1
                                    else:
                                        conv = 0
                                    if conv == 3:
                                        solution[i, j, k, m, 0] = day
                                        solution[i, j, k, m, 1] = usol[-1, 0]
                                        solution[i, j, k, m, 2] = usol[-1, 1]
                                        solution[i, j, k, m, 3] = usol[-1, 2]
                                
                            except:
                                print(f"Error while solving the system for mu_FD: {mu_FD}, mu_FS: {mu_FS}, mu_SF: {mu_SF}, mu_SF: {mu_SF} in day {day}")
                                errors.append([mu_FD, mu_FS, mu_DF, mu_SF, day])
                                break
            
            with open(f'3st_solution_{beta}.npy', 'wb') as file:
                np.save(file, solution);
            with open(f'3st_errors_{beta}.txt', 'a') as file:
                for line in errors:
                    file.write(f'{line}\n')
            clear()
            print(f"finished row mu_FD{i}")        
    t4 = time.time()
    print(f"experiment time : {t4 - t3}")