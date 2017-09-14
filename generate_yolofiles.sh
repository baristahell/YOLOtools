#!/bin/bash

# Set templates paths
template_cfg=/home/bastienhell/darknet/cfg/yolo-cityscapes-2.0_template.cfg
cfg_path=/home/bastienhell/darknet/cfg
template_data=/home/bastienhell/darknet/cfg/cityscapes_template.data
data_path=/home/bastienhell/darknet/cfg
train_file=/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/train_CS.txt
val_file=/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/val_CS.txt
template_names=/home/bastienhell/darknet/cfg/cityscapes.names
template_backup_folder=/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet/backup/cityscapes
template_results_folder=/home/bastienhell/darknet/results/cityscapes

#-------------------------------------------------------------------------------------- Set parameters 
#----------------------------------------------------------------------------- ONLY CHANGE THINGS HERE
# general
experience_name=13classes-10anchors-832x416-rand-t05
names_file=/home/bastienhell/darknet/cfg/cityscapes-13classes 	# default $template_names
width=832
height=416
# learning params
base_lr=0.0001 			# default 0.0001
max_batches=25000 		# default 45000
lr_steps=100,5000,15000 	# default 100,25000,35000
lr_scales=10,.1,.1 		# default 10,.1,.1
is_rand=1 			# default 0
thresh=0.5 			# default .6
# regions params
num_classes=13
num_anchors=10 
anchors_list=0.0752243585422,0.714258405934,0.179038706631,2.44557028826,0.285717553006,0.258765048349,0.417955698411,0.745602661603,0.568713915499,5.73069554826,0.821348082286,2.02536421038,1.02342667431,0.641234831721,1.97048956581,1.52414548333,3.83383514372,4.03242596644,13.0985830826,6.62819515049 # remove all spaces in list
# loss params
lambda_obj=5			# default 5 
lambda_noobj=1			# default 1
lambda_class=1			# default 1
lambda_coord=1			# default 1
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

# Generate everything !
backup_folder=$template_backup_folder/$experience_name
results_folder=$template_results_folder/$experience_name
new_cfg=$cfg_path/yolo-cityscapes-2.0-$experience_name.cfg
new_data=$data_path/cityscapes-$experience_name.data
names_file_sed=$(echo "$names_file" | sed 's/\//\\\//g')
train_file_sed=$(echo "$train_file" | sed 's/\//\\\//g')
val_file_sed=$(echo "$val_file" | sed 's/\//\\\//g')
backup_folder_sed=$(echo "$backup_folder" | sed 's/\//\\\//g')
results_folder_sed=$(echo "$results_folder" | sed 's/\//\\\//g')
num_filters=$((($num_classes+5)*$num_anchors))
mkdir $results_folder
mkdir $backup_folder
cp $template_cfg $new_cfg
cp $template_data $new_data
# .cfg 
sed -i "s/HEIGHT/$height/g" $new_cfg
sed -i "s/WIDTH/$width/g" $new_cfg
sed -i "s/BASELR/$base_lr/g" $new_cfg
sed -i "s/MAXBATCHES/$max_batches/g" $new_cfg
sed -i "s/LRSTEPS/$lr_steps/g" $new_cfg
sed -i "s/LRSCALES/$lr_scales/g" $new_cfg
sed -i "s/FILTERS/$num_filters/g" $new_cfg
sed -i "s/ANCHORSLIST/$anchors_list/g" $new_cfg
sed -i "s/NUMCLASSES/$num_classes/g" $new_cfg
sed -i "s/NUMANCHORS/$num_anchors/g" $new_cfg
sed -i "s/ISRAND/$is_rand/g" $new_cfg
sed -i "s/LAMBDAOBJ/$lambda_obj/g" $new_cfg
sed -i "s/LAMBDANOOBJ/$lambda_noobj/g" $new_cfg
sed -i "s/LAMBDACLASS/$lambda_class/g" $new_cfg
sed -i "s/LAMBDACOORD/$lambda_coord/g" $new_cfg
# .data
sed -i "s/NUMCLASSES/$num_classes/g" $new_data
sed -i "s/TRAINFILE/$train_file_sed/g" $new_data
sed -i "s/VALFILE/$val_file_sed/g" $new_data
sed -i "s/NAMESFILE/$names_file_sed/g" $new_data
sed -i "s/BACKUPFOLDER/$backup_folder_sed/g" $new_data
sed -i "s/RESULTSFOLDER/$results_folder_sed/g" $new_data
