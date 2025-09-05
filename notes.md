# 9.5
- trials
```bash
  # test run, detailed config saved at exp backup directory
  export PYTHONPATH=$(pwd)
  export CUDA_VISIBLE_DEVICES=0
  start_timestep=0 # start frame index for training
  end_timestep=-1 # end frame index, -1 for the last frame
  output_root="output"
  project="formal_training"
  expname="013-full-5cams-2"
  scene_idx=13
  dataset="chery/5cams"
  
  python tools/train.py \
      --config_file configs/streetgs.yaml \
      --output_root $output_root \
      --project $project \
      --run_name $expname \
      dataset=$dataset \
      data.scene_idx=$scene_idx \
      data.start_timestep=$start_timestep \
      data.end_timestep=$end_timestep
```

```bash
  # test run, detailed config saved at exp backup directory
  export PYTHONPATH=$(pwd)
  export CUDA_VISIBLE_DEVICES=0
  start_timestep=0 # start frame index for training
  end_timestep=-1 # end frame index, -1 for the last frame
  output_root="output"
  project="formal_training"
  expname="013-full-5cams-1"
  scene_idx=13
  dataset="chery/5cams"
  
  python tools/train.py \
      --config_file configs/streetgs.yaml \
      --output_root $output_root \
      --project $project \
      --run_name $expname \
      dataset=$dataset \
      data.scene_idx=$scene_idx \
      data.start_timestep=$start_timestep \
      data.end_timestep=$end_timestep
```

# Previous

```bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
start_timestep=0
end_timestep=-1
output_root="output"
project="formal_training"
expname="013-full-5cams"
scene_idx=13
dataset="chery/5cams"

python tools/train.py \ 
    --config_file configs/streetgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=$dataset \ 
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep 
```

```bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root="output"
project="default_project"
expname="nuscenes-000-all-3cams"
scene_idx=0
dataset="nuscenes/3cams"

python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=nuscenes/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```



# 8.19

- todo

  -  pay attention to lidar pose / ego pose on waymo dataset, current on chery it is ego pose 

- training board:

  - formal running - 5 cams

    - ```bash
      # test run, detailed config saved at exp backup directory
      export PYTHONPATH=$(pwd)
      export CUDA_VISIBLE_DEVICES=0
      start_timestep=0 # start frame index for training
      end_timestep=100 # end frame index, -1 for the last frame
      output_root="output"
      project="formal_training"
      expname="013-0-100-5cams"
      scene_idx=13
      dataset="chery/5cams"
      
      python tools/train.py \
          --config_file configs/streetgs.yaml \
          --output_root $output_root \
          --project $project \
          --run_name $expname \
          dataset=$dataset \
          data.scene_idx=$scene_idx \
          data.start_timestep=$start_timestep \
          data.end_timestep=$end_timestep
      ```

    - 

  - formal running - 3 cams

    - Problem: still floating gaussian, and getting worse and worse

    ```bash
    # test run, detailed config saved at exp backup directory
    export PYTHONPATH=$(pwd)
    export CUDA_VISIBLE_DEVICES=1
    start_timestep=0 # start frame index for training
    end_timestep=-1 # end frame index, -1 for the last frame
    output_root="output"
    project="unfixed_trial"
    expname="013"
    scene_idx=13
    start_timestep=0   # start frame index for training
    end_timestep=-1    # end frame index, -1 for the last frame
    dataset="chery/3cams"
    
    python tools/train.py \
        --config_file configs/streetgs.yaml \
        --output_root $output_root \
        --project $project \
        --run_name $expname \
        dataset=$dataset \
        data.scene_idx=$scene_idx \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep
    ```

    

  - ❌`output/unfixed_trial/012` only mainlidar

    - Dataset:`/mnt/public/AISIM/yuchen/repos/tools/chery_dataset_process/outputs_unfixed_lidarfixed/012`
    - most of the issues have been fixed
    - but only mainlidar

- Implementing

  - fix the problem of vice lidar
  - Ongoing:
    - 
    - under `/mnt/public/AISIM/yuchen/repos/tools/chery_dataset_process/outputs`
      - share directory: `sky_masks` & `fine_dynamic_masks`
      - 000:
        - more complex class name from original chery dataset annotation 
      - 001:
        - simpler class name of vehicle and pedestrian and cyclsit, aligned to drive studio.waymo
      - 002: 
        - fixed lidar colomn problem , add flow 6-9
      - 003：
        - without Boundingbox switching
      - 004:
        - make index consistent
        - Parallel the whole process
        - no axis changes for bounding box 
      - 005:
        - add axis changes for bounding box
  - what we have done to yuhan's code
    - extrinsics saving format -> need to revert ✅
    - saving name changed: lidar pose  -> need to revert ✅
    - dynamic masks ✅
    - fine dynamic masks ✅

- Run it : you don't know whether it is correct or not unless you try it directly on the final code 

  - Observing:

    - why the rendering of waymo's dynamix_masks is really simple, not a cub

  - Hard errors:

    - classname error when loading chery dataset as nuscene dataset
    - lidar inconsistency, waymo's are different to nuscene's

  - Script:

    - crazy run with yuhan's waymo + waymo's instances jsons

      ```bash
      # generate dynamic masks
      conda activate segformer
      segformer_path=SegFormer
      
      python datasets/tools/extract_masks.py \
          --data_root data/chery_yuchen \
          --segformer_path=$segformer_path \
          --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
          --scene_ids 000
      
      # test run 
      export PYTHONPATH=$(pwd)
      export CUDA_VISIBLE_DEVICES=1
      start_timestep=0 # start frame index for training
      end_timestep=-1 # end frame index, -1 for the last frame
      output_root="output"
      project="unfixed_trial"
      expname="013"
      scene_idx=13
      start_timestep=0   # start frame index for training
      end_timestep=-1    # end frame index, -1 for the last frame
      dataset="chery/3cams"
      
      python tools/train.py \
          --config_file configs/streetgs.yaml \
          --output_root $output_root \
          --project $project \
          --run_name $expname \
          dataset=$dataset \
          data.scene_idx=$scene_idx \
          data.start_timestep=$start_timestep \
          data.end_timestep=$end_timestep
      ```

      

    - process waymo dataset :
      dynamic scene 16,seg102319,0,-1,dynamic

      ```bash
      export PYTHONPATH=./
      
      python datasets/preprocess.py \
          --data_root data/waymo/raw/waymo_splits/dynamic32 \
          --target_dir data/waymo/processed \
          --dataset waymo \
          --split training \
          --scene_ids 16 \
          --workers 8 \
          --process_keys images lidar calib pose dynamic_masks objects
      ```

      

    - Trial run on chery dataset (failed due to chery is waymo format)

    ```bash
    export PYTHONPATH=$(pwd)
    export CUDA_VISIBLE_DEVICES=0
    start_timestep=0 # start frame index for training
    end_timestep=-1 # end frame index, -1 for the last frame
    output_root="output"
    project="default_project"
    expname="chery_tiral_01"
    scene_idx=0
    start_timestep=0   # start frame index for training
    end_timestep=-1    # end frame index, -1 for the last frame
    dataset="chery/3cams"
    
    python tools/train.py \
        --config_file configs/streetgs.yaml \
        --output_root $output_root \
        --project $project \
        --run_name $expname \
        dataset=$dataset \
        data.scene_idx=$scene_idx \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep
    ```

    

# 8.17

- chery to waymo dataset processing:

  - Problems:

    - saved json need obj_to_world position, while chery annotation provide self centered json

  - To check

    - yuhan's code provide static extrinsics (under going)
    - need to seperate dynamic masks （under going）
    - deal with width and height reverse problem

  - mask processor command

    - ```bash
      conda activate segformer
      segformer_path=SegFormer
      split=mini
      
      python datasets/tools/extract_masks.py \
          --data_root outputs/outputs \
          --segformer_path=$segformer_path \
          --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
          --start_idx 0 \
          --num_scenes 10 \
          --process_dynamic_mask
      ```

    - 

# 8.15

- large scale 3DGS with multiple GPU: https://www.alphaxiv.org/abs/2406.18533
- bash recording 这是一个中文测试 
- Train on drivestudio

```bash
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root="output"
project="default_project"
expname="nuscene_mini_03"
scene_idx=0
start_timestep=0   # start frame index for training
end_timestep=-1    # end frame index, -1 for the last frame
dataset="nuscenes/3cams"

python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=$dataset \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

- segformer process dataset:

```bash
conda activate segformer
segformer_path=SegFormer
split=mini

python datasets/tools/extract_masks.py \
    --data_root data/nuscenes/processed_10Hz/$split \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --start_idx 0 \
    --num_scenes 10 \
    --process_dynamic_mask
```

- Omnire provided processed data structure

```plain
ProjectPath/data/
  └── waymo/
    ├── raw/
    │    ├── segment-454855130179746819_4580_000_4600_000_with_camera_labels.tfrecord
    │    └── ...
    └── processed/
         └──training/
              ├── 001/
              │  ├──images/             # Images: {timestep:03d}_{cam_id}.jpg
              │  ├──lidar/              # LiDAR data: {timestep:03d}.bin
              │  ├──ego_pose/           # Ego vehicle poses: {timestep:03d}.txt
              │  ├──extrinsics/         # Camera extrinsics: {cam_id}.txt
              │  ├──intrinsics/         # Camera intrinsics: {cam_id}.txt
              │  ├──*sky_masks/          # Sky masks: {timestep:03d}_{cam_id}.png
              │  ├──*dynamic_masks/      # Coarse dynamic masks: category/{timestep:03d}_{cam_id}.png
              │  ├──*fine_dynamic_masks/ # (Optional) Fine dynamic masks: category/{timestep:03d}_{cam_id}.png 
              │  ├──*instances/          # Instances' bounding boxes information
              │  └──*humanpose/          # Preprocessed human body pose: smpl.pkl
              ├── 002/
              ├── ...
```

- <img src="/Users/xiyuchen/Library/Application Support/typora-user-images/image-20250815185234650.png" alt="image-20250815185234650" style="zoom:50%;" />

