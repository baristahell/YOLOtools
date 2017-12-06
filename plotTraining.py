import re,time
import matplotlib.pyplot as plt
import numpy as np

def avg_batch(batch):
    total_iou = 0
    for i in batch:
        total_iou += float(i)
    total_iou /= 8.0
    return total_iou

#-----------------------------------------------------------------# definitions
### get input/output
log_file = "/mnt/bigdrive/tempdata/caffetestdata/custom_dataset/data/backup/extractionweights/trainlog-custom_dataset-extractionweights" #"/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/trainlog-custom_dataset-sat1exp1"
### initiate lists
loss_list = []
iou_list = []
recall_list = []
file_lines = []
last_line = []
### init batch buffers
batchsize = 8
buffer_iou=[]
buffer_recall=[]

### set search patterns
loss_pattern = "\s([0-9]+\.[0-9]+)\savg"
iou_pattern = "IOU:\s([0-9]+\.[0-9]+)"
recall_pattern = "Avg Recall:\s([0-9]+\.[0-9]+)"
### open log
f = open(log_file, "r")

#-----------------------------------------------------------------# graph initialization
### initiate graph window
plt.ion()
fig = plt.figure(figsize=(15,10))
fig.suptitle(log_file)
plt.grid()
### loss graph
plt.subplot(3,1,1)
plt.grid()
ax_loss = fig.gca()
ax_loss.set_yticks(np.arange(0,1000,50))
plt.xlabel("batch")
plt.ylabel("loss")
### iou graph
plt.subplot(3,1,2)
plt.grid()
ax_iou = fig.gca()
ax_iou.set_yticks(np.arange(0,1.5,0.05))
ax_iou.tick_params(labeltop=False, labelright=True)
plt.xlabel("batch")
plt.ylabel("iou on batch")
### recall graph
plt.subplot(3,1,3)
plt.grid()
ax_recall = fig.gca()
ax_recall.set_yticks(np.arange(0,1.5,0.05))
ax_recall.tick_params(labeltop=False, labelright=True)
plt.xlabel("batch")
plt.ylabel("recall on batch")
### create plot curves
curve_loss, = ax_loss.plot(loss_list)
curve_iou, = ax_iou.plot(iou_list)
curve_recall, = ax_recall.plot(recall_list)

#-----------------------------------------------------------------# main loop
while(True):
    ### get file lines, buffer last one
    file_lines = f.read().split("\n")
    file_lines = np.append(last_line,file_lines)
    last_line = file_lines[-1]
    file_lines = file_lines[0:-2]
    ### find loss and iou instances
    for line in file_lines:
        ### read loss
        is_loss = re.search(loss_pattern, line)
        if is_loss:
            loss_list.append(is_loss.group(1))
        ### read iou
        is_iou = re.search(iou_pattern, line)
        if is_iou: 
            buffer_iou.append(is_iou.group(1))
            if len(buffer_iou)==batchsize:
                iou_list.append(buffer_iou[:])
                buffer_iou=[]       
        ### read recall
        is_recall = re.search(recall_pattern, line)
        if is_recall: 
            buffer_recall.append(is_recall.group(1))
            if len(buffer_recall)==batchsize:
                recall_list.append(buffer_recall[:])
                buffer_recall=[]

    ### update loss plot data
    t_loss = len(loss_list)
    x_loss = np.linspace(1,t_loss,t_loss)
    y_loss = loss_list
    curve_loss.set_data(x_loss,y_loss)
    ### update iou plot data
    t_iou = len(iou_list)
    x_iou = np.linspace(1,t_iou,t_iou)
    y_iou = map(lambda x:avg_batch(x),iou_list)
    curve_iou.set_data(x_iou,y_iou)
    ### update recall plot data
    t_recall = len(recall_list)
    x_recall = np.linspace(1,t_recall,t_recall)
    y_recall = map(lambda x:avg_batch(x),recall_list)
    curve_recall.set_data(x_recall,y_recall)

    ### update loss graph
    ax_loss.set_xlim(0,1.1*t_loss)
    ax_loss.set_ylim(0,1.1*np.array(y_loss,dtype=np.float).max())
    ax_loss.set_ylim(0,500)
    ### update iou graph 
    ax_iou.set_xlim(0,1.1*t_iou)
    ax_iou.set_ylim(0,1.1*np.array(y_iou,dtype=np.float).max())
    ### update recall graph 
    ax_recall.set_xlim(0,1.1*t_recall)
    ax_recall.set_ylim(0,1.1*np.array(y_recall,dtype=np.float).max())
    ### draw, prevent meltdown
    plt.draw()
    plt.pause(1)
    time.sleep(2)
