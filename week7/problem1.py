
# problem1.py
import pandas as pd
import numpy as np
import sys

def f_func(w,x_i):
    d = 2
    f_sum = 0
    for j in range(d+1):
        f_sum+=x_i.iloc[j] * w[j]
    f = 1 if f_sum > 0 else -1
    return f 

def perceptron(input_fileName,ouput_fileName):
    data = pd.read_csv(input_fileName,names=[0,1,3])
    d = 2
    data.insert (3, d, [1] * len(data))
    err = [1] * len(data)
    b = 0
    w_1 = 0
    w_2 = 0
    w = [w_1, w_2, b]
    output = []
    while sum(err) != 0:
        for i, example in data.iterrows():
            x_i = example[range(d+1)]
            y_i = example[d+1]
            err[i] = 0
            if y_i * f_func(w,x_i) <= 0:
                w +=  x_i * y_i
                err[i] = 1
                output.append(w[:])
    output_df = pd.DataFrame(output)
    output_df.to_csv(ouput_fileName, encoding='utf-8', index=False, header=False)

def main():
    input_fileName = sys.argv[1].lower()
    ouput_fileName = sys.argv[2].lower()
    perceptron(input_fileName, ouput_fileName)
        
if __name__ == '__main__':

    main()