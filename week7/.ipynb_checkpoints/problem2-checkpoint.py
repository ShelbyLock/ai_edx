# problem2.py
import pandas as pd
import numpy as np
import sys
def scalestd(data):
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0, ddof=0)
    data = (data - data_mean)/data_std
    return data

def getData(inputfileName):
    data = pd.read_csv(inputfileName,names=[1,2,3])
    n,_ = data.shape
    scaled_data = scalestd(data[[1,2]])
    scaled_data.insert (0, 0, [1] * n)
    scaled_data[3] = data[3]
    return scaled_data

def f(x_i, w):
    d = len(x_i)
    f_sum = 0
    f_sum=sum([x_i.iloc[j] * w[j] for j in range(d)])
    return f_sum

def descent(data, alpha, w):
    n,d = data.shape
    d = d - 1
    sumi = [0] * d
    for i, example in data.iterrows():
        indexs = list(range(d))
        x_i = example[indexs]
        y_i = example[d]
        diff = f(x_i, w) - y_i
        
        temp = (alpha/n) * diff
        sumi = [sumi[j] + temp * x_i.iloc[j] for j in range(d)] 
    return sumi

def err(data, w):
    n,d = data.shape
    d = d - 1
    sumi = 0
    for i, example in data.iterrows():
        indexs = list(range(d))
        x_i = example[indexs]
        y_i = example[d]
        diff = f(x_i, w) - y_i
        sumi+=diff
    sumi = sumi/2*n
    return sumi

def gradient_descent(alpha_set, n_iterations, data):
    n,d = data.shape
    d = d-1
    report=[]
    for alpha in alpha_set:
        print("Current Alpha:", alpha)
        err_set = []
        beta = [0] * d
        for i in range(n_iterations):
            descent_step = descent(data, alpha, beta)
            beta = [(beta[j] - descent_step[j]) for j in range(d)]
            err_set.append(abs(err(data, beta)))
        print("has minimon error in iteration: {}, the minimum error is {}, the last error value is {}".format(err_set.index(min(err_set)), min(err_set), err_set[-1]))          
        single_report = [alpha, n_iterations, beta[0],beta[1], beta[2]]
        report.append(single_report)
    return report

def main():
    input_fileName = sys.argv[1].lower()
    ouput_fileName = sys.argv[2].lower()
    #input_fileName = "input2.csv"
    #ouput_fileName = "output2.csv"
    scaled_data = getData(input_fileName)
    alpha_set = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    n_iterations = 100
    report = gradient_descent(alpha_set, n_iterations, scaled_data)
    alpha_set2 = [1.2]
    n_iterations = 32
    report.extend(gradient_descent(alpha_set2, n_iterations, scaled_data))
    report = pd.DataFrame(report)
    report.to_csv(ouput_fileName, encoding='utf-8', index=False, header=False)
    return report
if __name__ == '__main__':

    main()