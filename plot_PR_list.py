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

if __name__== "__main__":
    
    #----- get filenames
    file_0 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/vanilla/PRcurve_20000_withaddonval"
    file_1 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/thresh08/PRcurve_thresh08_final"
    #file_2 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/vanilla/PRcurve_900"
    #file_3 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/vanilla/PRcurve_500"
    #file_2 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/rand1/PRcurve_20000"
    #file_3 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/rand1/PRcurve_10000"
    #file_3 = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/results/cityscapes/13classes-10anchors-832x416-norand-t05/PRcurve"
    #file_2 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/thresh01/PRcurve_thresh01_10000"
    file_3 = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/results/sat1exp1/PRcurve_sat1exp1_final"
    files = [file_0,file_1,file_3]
    names= ["PRcurve_thresh05_20000_withaddonval","PRcurve_thresh08_final","PRcurve_sat1exp1_final"]
    
    #----- init results lists
    tre_list=[]
    rec_list=[]
    pre_list=[]
    f1score_list=[]

    #----- get results, append to lists
    for file_i in files:
        f = open(file_i,"r")
        results = f.readlines()
        tre,rec,pre=get_PR(results)
        tre_list.append(tre)
        rec_list.append(rec)
        pre_list.append(pre)
        f1score_list.append(get_f1score(tre,rec,pre))
        f.close()
        
    #----- plot graphs
    tre_ideal = [0,1,1]
    rec_ideal = [0,100,100]
    pre_ideal = [100,100,0]
    fsc_ideal = [100,100,0]
    fig = plt.figure()
    fig.suptitle("PR curves and f1 scores")
    # pr subplot
    plt.subplot(2,1,1)
    ax_pr = fig.gca()
    plt.xlim(0,110)
    plt.ylim(0,110)
    plt.grid()
    plt.xlabel("recall")
    plt.ylabel("precision")
    # f1score subplot
    plt.subplot(2,1,2)
    ax_fs = fig.gca()
    plt.xlim(0,1.1)
    plt.ylim(0,110)
    plt.grid()
    plt.xlabel("threshold")
    plt.ylabel("f1 score")
    # plot curves
    for i in range(0,len(tre_list)):
        ax_pr.plot(rec_list[i],pre_list[i],label=names[i])
        ax_fs.plot(tre_list[i],f1score_list[i],label=names[i])
    # plot ideal pr, legend and show
    ax_pr.plot(rec_ideal,pre_ideal,lw=3,label="ideal")
    ax_fs.plot(tre_ideal,fsc_ideal,lw=3,label="ideal")
    ax_pr.legend(loc='lower right',prop={'size':10})
    ax_fs.legend(loc='upper right',prop={'size':10})
    plt.show()
