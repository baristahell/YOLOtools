import re,time
import matplotlib.pyplot as plt
import numpy as np

##### definitions
### get input/output
log_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/trainlog-13classes-10anchors-832x416-cssteps"
# initiate lists
loss_list = []
iou_list = []
#recall_list = []
file_lines = []
last_line = []
### set search patterns
loss_pattern = "\s([0-9]+\.[0-9]+)\savg"
iou_pattern = "IOU:\s([0-9]+\.[0-9]+)"
#recall_pattern = "Avg Recall:\s([0-9]+\.[0-9]+)"
### open log
f = open(log_file, "r")

##### graph initialization
### initiate graph window
plt.ion()
fig = plt.figure(figsize=(10,10))
fig.suptitle(log_file)
plt.grid()
### loss graph
plt.subplot(2,1,1)
plt.grid()
ax_loss = fig.gca()
#ax_loss.set_xscale('log')
ax_loss.set_yticks(np.arange(0,1000,50))
plt.xlabel("batch")
plt.ylabel("loss")
### iou graph
plt.subplot(2,1,2)
plt.grid()
ax_iou = fig.gca()
#ax_iou.set_xscale('log')
ax_iou.set_yticks(np.arange(0,1.5,0.05))
plt.xlabel("image")
plt.ylabel("iou")
### create plot curves
curve_loss, = ax_loss.plot(loss_list)
curve_iou, = ax_iou.plot(iou_list)
### set smooth plot parameters
iou_smoother = 64 # size of the convolution filter for smoothing, set to 0 to skip

##### main loop
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
            iou_list.append(is_iou.group(1))
        ### read recall
        #is_recall = re.search(recall_pattern, line)
        #if is_recall:
        #    recall_list.append(is_recall.group(1))
    ### update loss plot data / graph
    t_loss = len(loss_list)
    x_loss = np.linspace(1,t_loss,t_loss)
    y_loss = loss_list
    curve_loss.set_data(x_loss,y_loss)
    ### update iou plot data
    t_iou = len(iou_list)
    if iou_smoother<=0:
        x_iou = np.linspace(1,t_iou,t_iou) # all iou points
        y_iou = iou_list
    else:
        smooth_iou = np.convolve(map(float,iou_list), np.ones((iou_smoother,))/iou_smoother, mode='valid') # smooth curve
        x_iou = np.linspace(1,len(smooth_iou),len(smooth_iou))
        y_iou = smooth_iou
    curve_iou.set_data(x_iou,y_iou)
    ### update loss graph
    ax_loss.set_xlim(0,1.1*t_loss)
    ax_loss.set_ylim(0,1.1*np.array(loss_list,dtype=np.float).max())
    ax_loss.set_ylim(0,800)
    fig.canvas.draw()
    plt.draw()
    ### update iou graph 
    if iou_smoother<=0:
        ax_iou.set_xlim(0,1.1*t_iou)
        ax_iou.set_ylim(0,1.1*np.array(iou_list,dtype=np.float).max()) # all iou points
    else:
        ax_iou.set_xlim(0,len(smooth_iou)*1.1)
        ax_iou.set_ylim(0,1.1*np.array(smooth_iou,dtype=np.float).max()) # smooth curve
    fig.canvas.draw()
    plt.draw()
    ### prevent meltdown
    time.sleep(3)
