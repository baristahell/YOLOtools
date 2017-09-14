import re
import matplotlib.pyplot as plt
import numpy as np


def get_PR(data):
    tre_pat = "Thresh:([0-9]+\.[0-9]+)"
    rec_pat = "Recall:([0-9]+\.[0-9]+)"
    pre_pat = "Precision:([0-9]+\.[0-9]+)"
    tre_list = []
    rec_list = []
    pre_list = []
    for line in data:
        is_tre = re.search(tre_pat,line)
        if is_tre:
            tre_list.append(is_tre.group(1))
        else:
            tre_list.append("NaN")
        is_rec = re.search(rec_pat,line)
        if is_rec:
            rec_list.append(is_rec.group(1))
        else:
            rec_list.append("NaN")
        is_pre = re.search(pre_pat,line)
        if is_pre:
            pre_list.append(is_pre.group(1))
        else:
            pre_list.append("NaN")
    return tre_list,rec_list,pre_list


if __name__ == "__main__":
    
    # open PRcurve results file (saved from terminal to txt or whatever)
    res_file ="test_read_PR"
    res_op = open(res_file,"r")
    results = res_op.readlines()
    res_op.close()

    # read PR data
    tre_list,rec_list,pre_list = get_PR(results)
      
    # plot PR curve and ideal
    rec_ideal = [0,100,100]
    pre_ideal = [100,100,0]
    rec_med = np.linspace(1,100,100)
    pre_med = 100-rec_med    
    plt.plot(rec_list,pre_list,'b',lw=2)
    plt.plot(rec_ideal,pre_ideal,'r',lw=4)
    plt.plot(rec_med,pre_med,'r--',lw=1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0,110)
    plt.ylim(0,110)
    plt.grid()
    plt.title("PR curve for "+ res_file)
    plt.show()
