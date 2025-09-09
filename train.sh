export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root="output"
project="default_project"
expname="test"

python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=chery/3cams \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep