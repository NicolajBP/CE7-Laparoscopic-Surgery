#!/bin/bash
#SBATCH --job-name=Tracking_Surgery
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24
#SBATCH --gres=gpu:1

start=$1
end=$2
tracker=$3
video_source=""
for i in $(seq $start $end); do
  formatted_value=$(printf "%02.0f" "$i")
  video_source+="cholec80_videos/video$formatted_value.mp4,"
done
echo $video_source

singularity exec --bind ~/CE_AVS_PBL:/CE_AVS_PBL /ceph/container/pytorch/pytorch_24.09.sif ../CE_AVS_PBL/bin/python3 ExtractLaparoscopicAPM.py --model_path 70.15.15WEIGHTS.pt --video_source $video_source --output_dir results --tracker $tracker --tracker_fps 1
