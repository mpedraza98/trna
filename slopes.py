import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from findiff import FinDiff, coefficients, Coefficient
# SET THE WORKING DIRECTORY
CWD = '/Users/miguel/Documents/Internship_CENTURI'
os.chdir(CWD)

## CREATES ARGUMENTS TO BE PARSED
parser = argparse.ArgumentParser(description="This script returns the slope for the linear part of the growth curve")
parser.add_argument(
    "-f",
    "--filename",
    dest = "filename",
    action="store",
    required = True,
    help="Uses the second derivative to determine the region where the plot is linear",
)
parser.add_argument(
    "-d",
    "--second-derivative",
    action="store_true",
    help="Uses the second derivative to determine the region where the plot is linear",
)
args = parser.parse_args()

## READS THE INPUT CSV FILE
FILENAME = args.filename
SAVE_FN = FILENAME.strip().split('.')[0]
START = 105 # Start of the linear portion of the graph
ALPH_DICT = {'A':0, 'B':1, 'C':2, 'D':3, 'E':3, 'F':5, 'G':6, 'H':7}
df = pd.read_csv(FILENAME)

## DEFINES FUNCTIONS FOR THE SCRIPT

def slopes_diff(data, epsilon = 6e-3, dx = 1):
    # Function takes as input the data an returns the slope of the largest linear segement
    # This segment is determined by the second derivative of the data being less than the threshold
    
    # Defines the element to calculate the second derivative
    dx = dx
    d_dx = FinDiff(0, dx, 2)
    d_temp = d_dx(data)
    
    # Test the threshold 
    temp_eps = np.argwhere(np.abs(d_temp) > epsilon).reshape(-1)
    if temp_eps[-1] != len(data) -1:
        temp_eps = np.append(temp_eps, len(data)-1)

    max_idx = np.argmax(np.diff(temp_eps))
    start, end = temp_eps[max_idx],  temp_eps[max_idx + 1]

    return [start, end]

# Creates a plot of the second derivative of the function
def plot_second_derivative(data, epsilon = 6e-3, dx = 1):
    dx = dx
    d_dx = FinDiff(0, 1, 2)
    plt.figure(figsize = (10,6))
    for i in df.columns[1:-1]:
        temp = df[i].values
        d_temp = d_dx(temp)
        plt.plot(d_temp, label = i)
    plt.hlines(6e-3, 0, 546, label = 'Threshold', linestyles='dashed')
    plt.xlabel('Timestep');
    plt.legend(loc ="upper right", fontsize="8");
    #plt.show();
    plt.savefig(SAVE_FN + '_second_derivative.png', dpi = 300)


# Creates a plot of the general behaviour of the growth curves

def plot_growth_curves(data, threshold = 105):
    plt.figure(figsize = (10,6))
    for i in df.columns[1:-1]:
        plt.plot(df[i], label = i)
        #plt.plot(df[i], marker = 'x', alpha = 0.5)
    plt.xticks(np.arange(0,501,100),list(df['Time'].values[np.arange(0,501,100)]))
    plt.vlines(threshold,0,1, linestyles='dashed')
    plt.xlabel('Time(HH:MM:SS)');
    plt.ylabel('OD');
    plt.title('Growth Curves');
    plt.legend(loc ="lower right", fontsize="8");
    plt.savefig(SAVE_FN + '_growth_curves.png', dpi = 300)

def lin_reg(data, derivative = False):
    '''
    data -> dataframe witht he information for bacterial growth
    '''
    temp_slope = {}
    plot_save = SAVE_FN + '_lin_reg_gc.png' if not derivative else SAVE_FN + '_diff_lin_reg_gc.png'

    X = np.arange(START, data.shape[0]).reshape(-1 , 1)
    plt.figure(figsize = (10 ,6))
    for i in data.columns[1:-1]:
        intervals = slopes_diff(df[i].values) if derivative else [START, data.shape[0]]
        temp_X = np.arange(intervals[0], intervals[1]).reshape(-1, 1)
        y = data[i].values[intervals[0]:intervals[1]]
        reg = LinearRegression().fit(temp_X, y)
        temp_slope[i] = [reg.coef_[0], reg.intercept_, reg.score(temp_X, y), intervals]
        plt.plot(data[i], alpha = 0.5)
        plt.plot(X, reg.predict(X), '--', label = i)

    plt.title('Linear fit for growth curves');
    plt.xlabel('Time(HH:MM:SS)');
    plt.ylabel('OD');
    plt.legend(loc ="lower right", fontsize="8");
    plt.savefig(plot_save, dpi = 300)

    return temp_slope

def heatmap_plot(data, slopes):
    grid = np.zeros((8,12))

    for i, j in enumerate(df.columns[1:-1]):
        row, col = ALPH_DICT[j[0]], int(j[1:])
        grid[row, col] = slopes[i]

    sns.set(rc={"figure.figsize":(10, 6)})
    ax = sns.heatmap(grid, linewidth = 0);
    ax.set_xticklabels(np.arange(1,13));
    ax.set_yticklabels(ALPH_DICT.keys());
    ax.xaxis.tick_top();
    ax.set_title(SAVE_FN);
    plt.savefig(SAVE_FN + "_heatmap.png", dpi = 300)



## CALL FUNCTIONS IMPLEMENTED ABOVE

plot_growth_curves(df)
lin_data = lin_reg(df)
coeff_df = pd.DataFrame(columns = ["slope", "intercept", "score", "intervals"], data = lin_data.values())



data_type = ['l'] * len(lin_data) 
if args.second_derivative:
    plot_second_derivative(df)
    diff_data = lin_reg(df, derivative = True)
    temp_df = pd.DataFrame(columns = coeff_df.columns , data = diff_data.values())
    coeff_df = coeff_df._append(temp_df)
    data_type += ['d'] * len(diff_data)
    heatmap_plot(df, temp_df.slope)


coeff_df['type'] = data_type

print(coeff_df)

plt.figure(figsize = (10,6))
plt.plot(coeff_df[coeff_df['type'] == 'l'].slope, label = 'Common interval')
plt.plot(coeff_df[coeff_df['type'] == 'd'].slope, label = 'Derivative')
plt.xticks(np.arange(0,len(lin_data)), df.columns[1:-1]);
plt.legend();
#plt.show();
plt.savefig(SAVE_FN + "_slopes_comparison.png", dpi = 300)