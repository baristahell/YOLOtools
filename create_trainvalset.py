from random import shuffle
import sys

# open train and val files, create list containing all imagenames
train_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/train_CS.txt"
val_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/val_CS.txt"
train = open(train_file,"r")
val = open(val_file,"r")
trainval_lines = train.read().split()+val.read().split()
train.close()
val.close()
# create trainval file joining train and val
out_trainval_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/trainval_CS.txt"
ftv = open(out_trainval_file, 'w')
ftv.write('\n'.join(trainval_lines))
ftv.close()
# shuffle lines
shuffle(trainval_lines)

# create extracted validation set 
val_lines_1 = trainval_lines[0:695]
val_lines_2 = trainval_lines[695:1390]
val_lines_3 = trainval_lines[1390:2085]
val_lines_4 = trainval_lines[2085:2780]
val_lines_5 = trainval_lines[2780:3475]

##print "1[0] : "+val_lines_1[0]+"   \t | 1[594] : "+val_lines_1[594]
##print "tv[0] : "+trainval_lines[0]+"    \t | tv[595] : "+trainval_lines[594]
##print "2[0] : "+val_lines_2[0]+"    \t | 2[594] : "+val_lines_2[594]
##print "tv[595] : "+trainval_lines[595]+"    \t | tv[1190] : "+trainval_lines[1189]
##print "3[0] : "+val_lines_3[0]+"    \t | 3[594] : "+val_lines_3[594]
##print "tv[1190] : "+trainval_lines[1190]+"    \t | tv[1785] : "+trainval_lines[1784]
##print "4[0] : "+val_lines_4[0]+"    \t | 4[594] : "+val_lines_4[594]
##print "tv[1785] : "+trainval_lines[1785]+"    \t | tv[2380] : "+trainval_lines[2379]
##print "5[0] : "+val_lines_5[0]+"    \t | 5[594] : "+val_lines_5[594]
##print "tv[2380] : "+trainval_lines[2380]+"    \t | tv[2975] : "+trainval_lines[2974]
##
##print len(val_lines_1)
##print len(val_lines_2)
##print len(val_lines_3)
##print len(val_lines_4)
##print len(val_lines_5)

# create training set accordingly
train_lines_1 = val_lines_2 + val_lines_3 + val_lines_4 + val_lines_5
train_lines_2 = val_lines_1 + val_lines_3 + val_lines_4 + val_lines_5
train_lines_3 = val_lines_1 + val_lines_2 + val_lines_4 + val_lines_5
train_lines_4 = val_lines_1 + val_lines_2 + val_lines_3 + val_lines_5
train_lines_5 = val_lines_1 + val_lines_2 + val_lines_3 + val_lines_4

# 
vals = [val_lines_1,val_lines_2,val_lines_3,val_lines_4,val_lines_5]
trains = [train_lines_1,train_lines_2,train_lines_3,train_lines_4,train_lines_5]

# write sets in corresponding files
for i in range(len(vals)):
    print str(i+1)
    out_val = vals[i]
    out_train = trains[i]
    out_val_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/val_CS_"+str(i+1)+".txt"
    out_train_file = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/train_CS_"+str(i+1)+".txt"
    
    fval = open(out_val_file, 'w')
    fval.write('\n'.join(out_val))
    fval.close()
    ftrain = open(out_train_file, 'w')
    ftrain.write('\n'.join(out_train))
    ftrain.close()
