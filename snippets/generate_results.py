import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    type_name = sys.argv[1]
    test_name = sys.argv[2]

    training = test_name
    target = "%s_test" % test_name
    #training_1 = "pert_random_1_testing_average"
    test = "%s_1_testing_average" % test_name



    df_training = pd.read_csv('data/%s.csv' % training, header=None)
    df_target = pd.read_csv('data/%s.csv' % target, header=None)

    #df_training_1 = pd.read_csv('results/%s.csv' % training_1, header=0)

    df_test = pd.read_csv('results/%s.csv' % test, header=0)

    df_all_frames = [df_training, df_target]
    df_all = pd.concat(df_all_frames)
    max_force = math.ceil(df_all[3].max())

    df_training_frames = [df_training]
    df_training = pd.concat(df_training_frames)
    df_training.columns = [type_name, 'horizontal_displacement', 'vertical_displacement', 'training_force']

    df_target_frames = [df_target]
    df_target = pd.concat(df_target_frames)
    df_target.columns = [type_name, 'horizontal_displacement', 'vertical_displacement', 'target_force']

    #df_training_frames = [df_training_1]
    #df_training = pd.concat(df_training_frames)
    #df_training.columns = ['pert', 'horizontal_displacement', 'vertical_displacement', 'target_force', 'training_force', 'test_error']

    df_test_frames = [df_test]
    df_test = pd.concat(df_test_frames)
    df_test.columns = [type_name, 'horizontal_displacement', 'vertical_displacement', 'target_force', 'test_force', 'test_error']

    font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}
    
    plt.figure(1)
    plt.rc('font', **font)
    plt.figure(figsize=(15,10))
    plt.ylim(ymin=0, ymax=max_force)
    plt.xlabel('Vertical Displacement')
    plt.ylabel('Force')
    training_plt = plt.scatter(df_training['vertical_displacement'], df_training['training_force'], color='lightcoral', s=25, alpha=1)
    target_plt = plt.scatter(df_target['vertical_displacement'], df_target['target_force'], color='darkred', s=25, alpha=1)
    #training_output = plt.scatter(df_training['vertical_displacement'], df_training['training_force'], color='g', s=5, alpha=0.5)
    test_plt = plt.scatter(df_test['vertical_displacement'], df_test['test_force'], color='b', s=5, alpha=1)
    legend = plt.legend(handles=[training_plt, target_plt, test_plt])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    legend.legendHandles[2]._sizes = [30]
    plt.savefig('results/{}/{}_v_summary.png'.format(type_name, test_name))

    plt.figure(1)
    plt.rc('font', **font)
    plt.figure(figsize=(15,10))
    plt.ylim(ymin=0, ymax=max_force)
    plt.xlabel('Horizontal Displacement')
    plt.ylabel('Force')
    training_plt = plt.scatter(df_training['horizontal_displacement'], df_training['training_force'], color='lightcoral', s=25, alpha=1)
    target_plt = plt.scatter(df_target['horizontal_displacement'], df_target['target_force'], color='darkred', s=25, alpha=1)
    #training_output = plt.scatter(df_training['vertical_displacement'], df_training['training_force'], color='g', s=5, alpha=0.5)
    test_plt = plt.scatter(df_test['horizontal_displacement'], df_test['test_force'], color='b', s=5, alpha=1)
    legend = plt.legend(handles=[training_plt, target_plt, test_plt])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    legend.legendHandles[2]._sizes = [30]
    plt.savefig('results/{}/{}_h_summary.png'.format(type_name, test_name))

if __name__ == "__main__":
    main()
