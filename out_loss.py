import re
import matplotlib.pyplot as plt
import numpy as np

# get loss and iou i/o
log_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/trainlog"

loss_res_file = "/home/bastienhell/darknet/results/cityscapes/loss"
iou_res_file = "/home/bastienhell/darknet/results/cityscapes/iou"
loss_list = []
iou_list = []

f = open(log_file, "r")
file_lines = f.readlines()
f.close()

# find loss and iou instances
loss_pattern = "\s([0-9]+\.[0-9]+)\savg"
iou_pattern = "IOU:\s([0-9]+\.[0-9]+)"
for line in file_lines:
    is_loss = re.search(loss_pattern, line)
    is_iou = re.search(iou_pattern, line)
    if is_loss:
        loss_list.append(is_loss.group(1))
    if is_iou:
        iou_list.append(is_iou.group(1))

# save loss and iou in files
res_loss = open(loss_res_file,"w")
res_loss.write("\n".join(loss_list))
res_loss.close

res_iou = open(iou_res_file,"w")
res_iou.write("\n".join(iou_list))
res_iou.close

# plot loss and iou
batch = np.linspace(1,len(loss_list),len(loss_list))
image_n = np.linspace(1,len(iou_list),len(iou_list))

##plt.plot(batch,loss_list)
##plt.xlabel("batch")
##plt.ylabel("loss")
##plt.grid()
##plt.show()

#smooth_iou = np.convolve(map(float,iou_list), np.ones((2560,))/2560, mode='valid')
plt.plot(iou_list)
plt.xlabel("image")
plt.ylabel("iou")
plt.grid()
plt.show()
