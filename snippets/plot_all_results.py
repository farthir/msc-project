import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

def main():
    input_filename = sys.argv[1]

    df = pd.read_csv('results/%s.csv' % input_filename).round(10)

    df.columns = ['input_0', 'input_1', 'input_2', 'target', 'test', 'testing_error']
    max_force = math.ceil(df.ix[:,4].max())
    
    font = {'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 14}

    rcParams.update({'figure.autolayout': True})
    
    plt.figure()
    plt.rc('font', **font)
    plt.ylim(ymin=0, ymax=max_force)
    plt.xlabel('Vertical Displacement (mm)')
    plt.ylabel('Force (kN)')
    outplt = plt.scatter(df['input_2'], df['target'], color='darkred', s=10, alpha=1)
    testoutplt = plt.scatter(df['input_2'], df['test'], color='c', s=2, alpha=1)
    legend = plt.legend(handles=[outplt, testoutplt])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    plt.savefig('results/%s_all_v.pdf' % input_filename)
    plt.show()

    plt.figure()
    plt.rc('font', **font)
    plt.ylim(ymin=0, ymax=max_force)
    plt.xlabel('Horizontal Displacement (mm)')
    plt.ylabel('Force (kN)')
    outplt = plt.scatter(df['input_1'], df['target'], color='darkred', s=10, alpha=1)
    testoutplt = plt.scatter(df['input_1'], df['test'], color='c', s=2, alpha=1)
    legend = plt.legend(handles=[outplt, testoutplt])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    plt.savefig('results/%s_all_h.pdf' % input_filename)
    plt.show()

if __name__ == "__main__":
    main()
