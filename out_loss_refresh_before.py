import re,time
import matplotlib.pyplot as plt
import numpy as np

# get loss i/o
log_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/trainlog"
loss_res_file = "/home/bastienhell/darknet/results/cityscapes/loss"
loss_list = []
loss_list_full = []

f = open(log_file, "r")

# create graph window
plt.ion()
fig = plt.figure()
ax = fig.gca()
plt.xlabel("batch")
plt.ylabel("loss")
plt.grid()
curve, = ax.plot(loss_list)
    
# always loop
while(True):
    file_lines = f.readlines()
    #f.close()

    # find loss and iou instances
    loss_pattern = "\s([0-9]+\.[0-9]+)\savg"
    for line in file_lines:
        is_loss = re.search(loss_pattern, line)
        if is_loss:
            loss_list.append(is_loss.group(1))

    loss_list_full = np.append(loss_list,loss_list_full)
    #loss_list_full.append(loss_list)
    
    # save loss in files
    #res_loss = open(loss_res_file,"w")
    #res_loss.write("\n".join(loss_list))
    #res_loss.close
    
    #print "len(loss_list): " + str(len(loss_list))
    #print "len(loss_list_full): " + str(len(loss_list_full))

    x = np.linspace(1,len(loss_list),len(loss_list))
    #print len(x)
    y = loss_list
    #print len(y)
    curve.set_data(x,y)

    ax.set_xlim(0,1.1*len(loss_list))
    ax.set_ylim(0,1.1*np.array(loss_list,dtype=np.float).max())
    
    #print loss_list_full.max()
    fig.canvas.draw()
    #plt.draw()


    time.sleep(1)

#smooth_loss = np.convolve(map(float,loss_list), np.ones((64,))/64, mode='valid')
