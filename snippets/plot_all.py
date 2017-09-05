import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm

def main():
    input_filename = sys.argv[1]

    df = pd.read_csv('data/%s.csv' % input_filename).round(10)

    df.columns = ['property', 'horizontal', 'vertical', 'force']
    #max_force = math.ceil(df.ix[:,3].max())
    
    font = {'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 14}

    rcParams.update({'figure.autolayout': True})
    properties = np.unique(df["property"])

    plt.figure()
    plt.rc('font', **font)
    plt.xlabel('Vertical Displacement (mm)')
    plt.ylabel('Force (kN)')
    for prop in properties:
        label = '$\\bar{\\rho} = %s$' % prop
        plt_df = df[df['property'] == prop]
        plt.scatter(plt_df['vertical'], plt_df['force'], label=label, s=10, c=cm.Set1(prop))

    legend = plt.legend()
    plt.savefig('data/%s_v.pdf' % input_filename)
    plt.show()
    
    plt.figure()
    plt.rc('font', **font)
    plt.xlabel('Horizontal Displacement (mm)')
    plt.ylabel('Force (kN)')
    for prop in properties:
        label = '$\\bar{\\rho} = %s$' % prop
        plt_df = df[df['property'] == prop]
        plt.scatter(plt_df['horizontal'], plt_df['force'], label=label, s=10, c=cm.Set1(prop))

    legend = plt.legend()
    plt.savefig('data/%s_h.pdf' % input_filename)
    plt.show()

if __name__ == "__main__":
    main()
