import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
    'size'   : 14}

    rcParams.update({'figure.autolayout': True})

    plt.figure()
    plt.rc('font', **font)
    plt.xlabel('Strain')
    plt.ylabel('Stress (MPa)')
    plt.xlim(xmin=0, xmax=max_strain)
    plt.ylim(ymin=-max_stress*0.1, ymax=max_stress)

    plt.scatter(df_training['strain'], df_training['stress'], color='darkred', s=5, alpha=1)
    plt.savefig('results/{}/{}_stress_strain.pdf'.format(type_name, test_name))

if __name__ == "__main__":
    main()
