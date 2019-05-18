import argparse
import os
import subprocess
import logging


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="the name of the yml config file to run. For example: configs/default.yml")

    fmt = "[%(levelname)8s |%(funcName)21s:%(lineno)3d]   %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
    )

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
#SBATCH --output=logs/{content_img}_%a.log
#SBATCH --mem=30g
#SBATCH --time=01:00:00
#SBATCH --partition gpu
#SBATCH --array=1-{len(style_files)}
#SBATCH -J {content_img}

module load compilers/cuda/9.2
. ~/miniconda/etc/profile.d/conda.sh
conda activate style

PARAMS=`expr ${{SLURM_ARRAY_TASK_ID}} - 1`
List={' '.join(style_files)}
arr=($List)
style=${{arr[$PARAMS]}}
echo $style

startt=$(date +"%T")
python neural_style.py --content_img {content_img} --content_img_dir {content_img_dir} --style_imgs $style --max_size 1024 --max_iterations 1000 --device /gpu:0 --verbose
endt=$(date +"%T")
echo "Start to finish time: $startt -> $endt"
    """
    logging.info("Writing slurm job script")
    slurm_filename = os.path.join("jobs", content_img.split(".")[0] + ".job")
    with open(slurm_filename, "w") as f:
        f.write(template)
    logging.info("Executing sbatch")
    subprocess.run(["sbatch", slurm_filename])
    logging.info("sbatch executed")
