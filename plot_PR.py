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
            if float(is_pre.group(1))>100:
                pre_list.append(100)
            else:
                pre_list.append(is_pre.group(1))
        else:
            pre_list.append("NaN")
    return tre_list,rec_list,pre_list

def get_f1score(tre,rec,pre):
    f1score=[]
    for i in range(0,len(tre)):
        if np.isnan(float(rec[i])) or np.isnan(float(pre[i])) or float(rec[i])==0 and float(pre[i])==0:
            f1score.append("NaN")
        else:
            f1score.append(2*float(rec[i])*float(pre[i])/(float(rec[i])+float(pre[i])))
    return f1score

if __name__ == "__main__":
    
    # open PRcurve results file (saved from terminal to txt or whatever)
    #res_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/results/cityscapes/13classes-10anchors-832x416-norand-t05/PRcurve"
    res_file = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/vanilla/PRcurve_20000"
    res_op = open(res_file,"r")
    results = res_op.readlines()
    res_op.close()

    #res_file2 = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/results/cityscapes/13classes-10anchors-832x416-norand-t05/PRcurve"
    res_file2 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/rand1/PRcurve_rand1-final"
    res_op2 = open(res_file2,"r")
    results2 = res_op2.readlines()
    res_op2.close()
    
    # read PR data
    tre_list,rec_list,pre_list = get_PR(results)
    tre_list2,rec_list2,pre_list2 = get_PR(results2)
    
    # plot PR curve and ideal
    rec_ideal = [0,100,100]
    pre_ideal = [100,100,0]
    rec_med = np.linspace(1,100,100)
    pre_med = 100-rec_med
    plt.subplot(121)
    custom_vanilla, = plt.plot(rec_list,pre_list,'b-',lw=1, label="custom_vanilla")
    custom_rand, = plt.plot(rec_list2,pre_list2,'g-',lw=1, label="custom_rand")
    ideal, = plt.plot(rec_ideal,pre_ideal,'r',lw=3, label="ideal")
    plt.plot(rec_med,pre_med,'r--',lw=1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim(0,110)
    plt.ylim(0,110)
    plt.grid()
    #plt.title("PR curve for "+ res_file)
    plt.title("Precision-Recall curves")
    plt.legend(handles=[custom_vanilla, custom_rand, ideal])
    #plt.show()

    f1score1=get_f1score(tre_list,rec_list,pre_list)
    f1score2=get_f1score(tre_list2,rec_list2,pre_list2)
    
    
    plt.subplot(122)
    plt.title("F1-score curves")
    plt.xlabel("threshold")
    plt.ylabel("f1-score")
    plt.xlim(0,1)
    plt.ylim(0,100)
    plt.grid()
    plt.plot(tre_list,f1score1,'b-')
    plt.plot(tre_list2,f1score2,'g-')
    plt.show()
