import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    input_filename = sys.argv[1]
    num_networks = int(sys.argv[2])

    df = pd.read_csv('results/%s.csv' % input_filename).round(10)

    new_df = pd.DataFrame(dtype=float)

    duplicate = df.duplicated(('input_0', 'input_1', 'input_2', 'output_0'))

    for index, row in df.iterrows():
        if duplicate[index] == True:
            continue

        same_rows = df[
            (df['input_0'] == row['input_0']) &
            (df['input_1'] == row['input_1']) &
            (df['input_2'] == row['input_2']) &
            (df['output_0'] == row['output_0'])
        ]

        num_rows = int(same_rows.size / 6)

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

    plt.xlabel('Horizontal Displacement')
    plt.ylabel('Force')
    plt.scatter(new_df['input_1'], new_df['output_0'], color='r', s=1)
    plt.scatter(new_df['input_1'], new_df['test_output_0'], color='g', s=1)
    plt.ylim(ymin=0)
    plt.savefig('results/%s_average.svg' % input_filename)
    plt.show()

if __name__ == "__main__":
    main()
