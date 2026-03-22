#!/bin/bash
cd ../baselines/siren


### Scripts to evaluate the siren model on different BDD splits on multiple GPUs
# process_eval_dataset() {
#     local split=$1
#     local dataset_file="BDD-standard_${split}.hdf5"
#     cp /mnt/hdd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/ViTDET/${dataset_file} /home/khoadv/SAFE/SAFE_Official/dataset_dir/safe/ViTDET
#     python siren.py --variant ViTDET --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate --i-split-for-training $split > ./log/ViTDET_siren_bdd_coco_layer_features_seperate_split_${split}.txt 2>&1
#     python siren.py --variant ViTDET --dataset-name bdd --ood-dataset-name openimages --osf-layers layer_features_seperate --i-split-for-training $split > ./log/ViTDET_siren_bdd_openimages_layer_features_seperate_split_${split}.txt 2>&1
#     rm /home/khoadv/SAFE/SAFE_Official/dataset_dir/safe/ViTDET/${dataset_file}
# }
# process_eval_dataset 0
# process_eval_dataset 1
# process_eval_dataset 2
# process_eval_dataset 3
# process_eval_dataset 4
# process_eval_dataset 5
# process_eval_dataset 6


### Scripts to train the siren model on VOC on multiple GPUs

# # Function to run a process on a specific GPU
# run_on_gpu() {
#     local gpu_id=$1
#     local start_idx_layer=$2
#     local end_idx_layer=$3
#     CUDA_VISIBLE_DEVICES=$gpu_id python siren.py --variant ViTDET --dataset-name voc --ood-dataset-name coco --osf-layers layer_features_seperate --start-idx-layer $start_idx_layer --end-idx-layer $end_idx_layer > ./log/ViTDET_siren_voc_coco_layer_features_seperate_train_${start_idx_layer}_${end_idx_layer}.txt 2>&1 &
#     echo $! > ./log/process_${start_idx_layer}.pid
# }

# # Function to process a dataset
# process_dataset() {
    
#     # Run four processes on different GPUs
#     run_on_gpu 0 0 70
#     run_on_gpu 1 70 140
#     run_on_gpu 2 140 210
#     run_on_gpu 3 210 400

#     # Wait for all processes to complete
#     echo "Waiting for all processes to complete..."
#     for start_idx_layer in 0 70 140 210; do
#         if [ -f ./log/process_${start_idx_layer}.pid ]; then
#             wait $(cat ./log/process_${start_idx_layer}.pid)
#             rm ./log/process_${start_idx_layer}.pid
#         fi
#     done
#     echo "All processes completed!"
    
# }
# process_dataset


### Scripts to train the siren model on BDD difference splits on multiple GPUs

# Function to run a process on a specific GPU
run_on_gpu() {
    local gpu_id=$1
    local split=$2
    local start_idx_layer=$3
    local end_idx_layer=$4
    CUDA_VISIBLE_DEVICES=$gpu_id python siren.py --variant ViTDET --dataset-name bdd --ood-dataset-name coco --osf-layers layer_features_seperate --i-split-for-training $split --start-idx-layer $start_idx_layer --end-idx-layer $end_idx_layer > ./log/ViTDET_siren_bdd_coco_layer_features_seperate_train_split_${split}_${start_idx_layer}_${end_idx_layer}.txt 2>&1 &
    echo $! > ./log/process_${start_idx_layer}.pid
}

# Function to process a dataset split
process_dataset_split() {
    local split=$1
    local dataset_file="BDD-standard_${split}.hdf5"
    
    # Copy the dataset file
    scp khoadv@172.31.123.20:/mnt/hdd/khoadv/Backup/SAFE/dataset_dir/safe/Object_Specific_Features/All_Osf_Layers_Features/ViTDET/${dataset_file} /home/khoa/SAFE/SAFE_Modified/dataset_dir/safe/ViTDET

    # Run four processes on different GPUs
    run_on_gpu 0 $split 0 8
    run_on_gpu 1 $split 8 16
    run_on_gpu 2 $split 16 24
    run_on_gpu 3 $split 24 35

    # Wait for all processes to complete
    echo "Waiting for all processes to complete..."
    for start_idx_layer in 0 8 16 24; do
        if [ -f ./log/process_${start_idx_layer}.pid ]; then
            wait $(cat ./log/process_${start_idx_layer}.pid)
            rm ./log/process_${start_idx_layer}.pid
        fi
    done
    echo "All processes completed!"
    
    # Clean up the dataset file
    rm /home/khoa/SAFE/SAFE_Modified/dataset_dir/safe/ViTDET/${dataset_file}
}

# Process each dataset split
# process_dataset_split 0
# process_dataset_split 1
# process_dataset_split 2
# process_dataset_split 3
# process_dataset_split 4
# process_dataset_split 5
# process_dataset_split 6
