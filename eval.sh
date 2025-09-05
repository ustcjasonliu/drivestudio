export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame
output_root="output"
project="default_project"
expname="chery-3cams"

python tools/eval.py --resume_from /mnt/public/jason/drivestudio1/output/default_project/chery-3cams/checkpoint_final.pth