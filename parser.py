import argparse
import os
import subprocess

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="the name of the yml config file to run. For example: configs/default.yml")

    args = parser.parse_args()
    image_path = args.image
    content_img_dir = os.path.dirname(image_path)
    content_img = os.path.basename(image_path)

    style_files = os.listdir("styles")

    os.makedirs("logs", exist_ok=True)
    os.makedirs("jobs", exist_ok=True)

    template = f"""#!/bin/bash
#SBATCH --ntasks=1 # Run 1 task
#SBATCH --nodes=1 # Run the task on a single node
#SBATCH --gres=gpu:1 # Request both GPUs
#SBATCH --cpus-per-task=1 # Request 2 CPUs
#SBATCH --output=logs/output_%a.log
#SBATCH --mem=30g
#SBATCH --time=01:00:00
#SBATCH --partition gpu
#SBATCH --array=1-{len(style_files)}

module load compilers/cuda/9.2
. ~/miniconda/etc/profile.d/conda.sh
conda activate style

PARAMS=`expr ${{SLURM_ARRAY_TASK_ID}} - 1`
List={' '.join(style_files)}
arr=($List)

startt=$(date +"%T")
python neural_style.py --content_img {content_img} --content_img_dir {content_img_dir} --style_imgs ${{arr[$PARAMS]}} --max_size 1024 --max_iterations 1000 --device /gpu:0 --verbose
endt=$(date +"%T")
echo "Start to finish time: $startt -> $endt"
    """
    slurm_filename = os.path.join("jobs", content_img.split(".")[0] + ".job")
    with open(slurm_filename, "w") as f:
        f.write(template)

    subprocess.run(["sbatch", slurm_filename], shell=True)
