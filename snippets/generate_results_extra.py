import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def main():
    type_name = sys.argv[1]
    test_name = sys.argv[2]
    test_number = sys.argv[3]

    training = test_name
    target = "%s_test" % test_name
    #training_1 = "pert_random_1_testing_average"
    test = "{}_{}_testing_average".format(test_name, test_number)

    df_training = pd.read_csv('data/%s.csv' % training, header=None)
    df_target = pd.read_csv('data/%s.csv' % target, header=None)

    #df_training_1 = pd.read_csv('results/%s.csv' % training_1, header=0)

    df_test = pd.read_csv('results/%s.csv' % test, header=0)

    df_all_frames = [df_training, df_target]
    df_all = pd.concat(df_all_frames)
    max_force = math.ceil(df_all[3].max())

    df_training_frames = [df_training]
    df_training = pd.concat(df_training_frames)
    df_training.columns = [type_name, 'horizontal_displacement', 'vertical_displacement', 'training']

    df_target_frames = [df_target]
    df_target = pd.concat(df_target_frames)
    df_target.columns = [type_name, 'horizontal_displacement', 'vertical_displacement', 'target']

    #df_training_frames = [df_training_1]
    #df_training = pd.concat(df_training_frames)
    #df_training.columns = ['pert', 'horizontal_displacement', 'vertical_displacement', 'target', 'training', 'test_error']

    df_test_frames = [df_test]
    df_test = pd.concat(df_test_frames)
    df_test.columns = [type_name, 'horizontal_displacement', 'vertical_displacement', 'target', 'test', 'test_error']

    training_types = np.unique(df_training[type_name])

    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

    rcParams.update({'figure.autolayout': True})
    
    plt.figure()
    plt.rc('font', **font)
    #plt.figure(figsize=(15,10))
    plt.ylim(ymin=0, ymax=max_force)
    plt.xlabel('Vertical Displacement (mm)')
    plt.ylabel('Force (kN)')
    for training_type in training_types:
        plt_df = df_training[df_training[type_name] == training_type]
        if (training_type*10) % 1 == 0:
            training_plt = plt.scatter(plt_df['vertical_displacement'], plt_df['training'], color='lightcoral', s=4, alpha=1)
        else:
            training_extra_plt = plt.scatter(plt_df['vertical_displacement'], plt_df['training'], color='burlywood', label='training_extra', s=4, alpha=1)
    target_plt = plt.scatter(df_target['vertical_displacement'], df_target['target'], color='darkred', s=4, alpha=1)
    #training_output = plt.scatter(df_training['vertical_displacement'], df_training['training'], color='g', s=5, alpha=0.5)
    test_plt = plt.scatter(df_test['vertical_displacement'], df_test['test'], color='c', s=2, alpha=1)
    legend = plt.legend(handles=[training_plt, training_extra_plt, target_plt, test_plt], loc=2, framealpha=0.5)
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    legend.legendHandles[2]._sizes = [30]
    legend.legendHandles[3]._sizes = [30]
    plt.savefig('results/{}/{}_{}_v_summary.pdf'.format(type_name, test_name, test_number))

    plt.figure()
    plt.rc('font', **font)
    #plt.figure(figsize=(15,10))
    plt.ylim(ymin=0, ymax=max_force)
    plt.xlabel('Horizontal Displacement (mm)')
    plt.ylabel('Force (kN)')
    for training_type in training_types:
        plt_df = df_training[df_training[type_name] == training_type]
        if (training_type*10) % 1 == 0:
            training_plt = plt.scatter(plt_df['horizontal_displacement'], plt_df['training'], color='lightcoral', s=4, alpha=1)
        else:
            training_extra_plt = plt.scatter(plt_df['horizontal_displacement'], plt_df['training'], color='burlywood', label='training_extra', s=4, alpha=1)
    target_plt = plt.scatter(df_target['horizontal_displacement'], df_target['target'], color='darkred', s=4, alpha=1)
    #training_output = plt.scatter(df_training['vertical_displacement'], df_training['training'], color='g', s=5, alpha=0.5)
    test_plt = plt.scatter(df_test['horizontal_displacement'], df_test['test'], color='c', s=2, alpha=1)
    legend = plt.legend(handles=[training_plt, training_extra_plt, target_plt, test_plt], loc=2, framealpha=0.5)
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    legend.legendHandles[2]._sizes = [30]
    legend.legendHandles[3]._sizes = [30]
    plt.savefig('results/{}/{}_{}_h_summary.pdf'.format(type_name, test_name, test_number))

if __name__ == "__main__":
    main()
