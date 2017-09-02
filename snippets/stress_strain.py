import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

def main():
    type_name = sys.argv[1]
    test_name = sys.argv[2]

    training = test_name

    df_training = pd.read_csv('data/%s.csv' % training, header=None)
    df_training.columns = [type_name, 'horizontal_displacement', 'vertical_displacement', 'training_force']

    df_training['strain'] = df_training['vertical_displacement'] / 52.56
    df_training['stress'] = df_training['training_force'] * 1000 / 57.33 / 10

    max_strain = df_training['strain'].max() * 1.1
    max_stress = df_training['stress'].max() * 1.1

    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 20}

    plt.figure(1)
    plt.rc('font', **font)
    plt.figure(figsize=(15, 10))
    plt.xlim(xmin=0, xmax=max_strain)
    plt.ylim(ymin=-max_stress*0.1, ymax=max_stress)
    plt.xlabel('Strain', fontsize=24)
    plt.ylabel('Stress (MPa)', fontsize=24)
    plt.scatter(df_training['strain'], df_training['stress'], color='b', s=25, alpha=1)
    plt.savefig('results/{}/{}_stress_strain.png'.format(type_name, test_name))

if __name__ == "__main__":
    main()
