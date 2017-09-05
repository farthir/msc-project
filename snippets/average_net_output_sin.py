import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

def main():
    input_filename = sys.argv[1]
    num_networks = int(sys.argv[2])

    df = pd.read_csv('results/%s.csv' % input_filename).round(10)

    new_df = pd.DataFrame(dtype=float)

    duplicate = df.duplicated(('input_0', 'output_0'))

    for index, row in df.iterrows():
        if duplicate[index] == True:
            continue

        same_rows = df[
            (df['input_0'] == row['input_0']) &
            (df['output_0'] == row['output_0'])
        ]

        num_rows = int(same_rows.size / 4)

        if num_rows != num_networks:
            print("WARN: row index {} has {} iterations when {} were expected".format(index, num_rows, num_networks))
            print(row)
        
        avg_test_output_0 = same_rows['test_output_0'].mean()
        avg_testing_error = same_rows['testing_error'].mean()
        row = same_rows.iloc[0]
        row.set_value('test_output_0', avg_test_output_0)
        row.set_value('testing_error', avg_testing_error)
        new_df = new_df.append(row, ignore_index=True)

    new_df.to_csv('results/%s_average.csv' % input_filename, index=False)
    new_df.columns = ['input_0', 'target', 'test', 'error']
    
    font = {'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 14}

    rcParams.update({'figure.autolayout': True})
    
    plt.figure()
    plt.rc('font', **font)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    outplt = plt.scatter(new_df['input_0'], new_df['target'], color='darkred', s=10, alpha=1)
    testoutplt = plt.scatter(new_df['input_0'], new_df['test'], color='c', s=5, alpha=1)
    legend = plt.legend(handles=[outplt, testoutplt])
    legend.legendHandles[0]._sizes = [30]
    legend.legendHandles[1]._sizes = [30]
    plt.savefig('results/%s_average.pdf' % input_filename)
    plt.show()

if __name__ == "__main__":
    main()
