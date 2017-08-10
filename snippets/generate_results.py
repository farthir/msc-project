import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    test = sys.argv[1]

    target_1 = "pert_00"
    target_2 = "pert_00_test"
    training_1 = "pert_random_1_testing_average"
    test_1 = "%s_1_testing_average" % test

    df_target_1 = pd.read_csv('data/%s.csv' % target_1, header=None)
    df_target_2 = pd.read_csv('data/%s.csv' % target_2, header=None)

    df_training_1 = pd.read_csv('results/%s.csv' % training_1, header=0)

    df_test_1 = pd.read_csv('results/%s.csv' % test_1, header=0)

    df_target_frames = [df_target_1, df_target_2]
    df_target = pd.concat(df_target_frames)
    df_target.columns = ['pert', 'horizontal_displacement', 'vertical_displacement', 'target_force']

    df_training_frames = [df_training_1]
    df_training = pd.concat(df_training_frames)
    df_training.columns = ['pert', 'horizontal_displacement', 'vertical_displacement', 'target_force', 'training_force', 'test_error']

    df_test_frames = [df_test_1]
    df_test = pd.concat(df_test_frames)
    df_test.columns = ['pert', 'horizontal_displacement', 'vertical_displacement', 'target_force', 'test_force', 'test_error']

    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}
    
    plt.figure(1)
    plt.rc('font', **font)
    plt.figure(figsize=(15,10))
    plt.ylim(ymin=0, ymax=1.4)
    plt.xlabel('Vertical Displacement')
    plt.ylabel('Force')
    target_output = plt.scatter(df_target['vertical_displacement'], df_target['target_force'], c=df_target['pert'], cmap='copper', s=25, alpha=1, marker='x')
    training_output = plt.scatter(df_training['vertical_displacement'], df_training['training_force'], color='b', s=5, alpha=1)
    test_output = plt.scatter(df_test['vertical_displacement'], df_test['test_force'], color='g', s=25, alpha=1, marker='+')
    legend = plt.legend(handles=[target_output, training_output, test_output])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[0].set_color('red')
    legend.legendHandles[1]._sizes = [30]
    legend.legendHandles[2]._sizes = [30]
    plt.savefig('results/pert/%s_v_summary.png' % test)

    plt.figure(2)
    plt.rc('font', **font)
    plt.figure(figsize=(15,10))
    plt.ylim(ymin=0, ymax=1.4)
    plt.xlabel('Horizontal Displacement')
    plt.ylabel('Force')
    target_output = plt.scatter(df_target['horizontal_displacement'], df_target['target_force'], c=df_target['pert'], cmap='copper', s=25, alpha=1, marker='x')
    training_output = plt.scatter(df_training['horizontal_displacement'], df_training['training_force'], color='g', s=5, alpha=1)
    test_output = plt.scatter(df_test['horizontal_displacement'], df_test['test_force'], color='b', s=25, alpha=1, marker='+')
    legend = plt.legend(handles=[target_output, training_output, test_output])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[0].set_color('red')
    legend.legendHandles[1]._sizes = [30]
    legend.legendHandles[2]._sizes = [30]
    plt.savefig('results/pert/%s_h_summary.png' % test)

if __name__ == "__main__":
    main()
