#!/encs/bin/tcsh

#SBATCH --job-name=GMamabIR_train          # Job name
#SBATCH --output=./logs/%x_%j.out             # Standard output and error log
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks=1                           # Number of tasks (usually 1 for GPU jobs)
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1
#SBATCH --mem=40G                    # Adjust memory as needed


echo "Starting SLURM job script..."

# Purge any loaded modules to start clean
module purge

# Load necessary modules
module load gcc/9.3
module load anaconda3/2023.03/default

# Since the node has CUDA 12.4 installed, we don't need to load an incompatible CUDA version>
echo "Modules loaded successfully."


# Activate the conda environment
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
if ($status != 0) then
  echo "Failed to activate the Conda environment."
  exit 1
endif

cd /speed-scratch/se_gham/GatedMambaIR

echo "Conda initialized successfully."

# Debugging outputs
echo "Current working directory: `pwd`"
echo "Checking available Python versions:"
which python
python --version

# Write the Python commands to a temporary script
set temp_script = "check_cuda.py"
cat << EOF > $temp_script

import torch

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected")
EOF

# Execute the Python script
echo "Checking CUDA availability and GPU details from PyTorch:"
python $temp_script

# Clean up temporary script without confirmation
rm -f $temp_script

if ($status != 0) then
  echo "PyTorch could not detect CUDA correctly. Please check your installation."
  exit 1
endif

# Check CUDA Version
echo "---- CUDA Version ----"
python -c "import torch; print('Torch CUDA Version:', torch.version.cuda)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA Device Count:', torch.cuda.device_count())"

# Check nvidia-smi
echo "---- NVIDIA-SMI Output ----"
nvidia-smi

setenv HF_HOME /speed-scratch/se_gham/hf_cache
setenv TRANSFORMERS_CACHE /speed-scratch/se_gham/hf_cache
setenv HF_DATASETS_CACHE /speed-scratch/se_gham/hf_cache/datasets
setenv HF_METRICS_CACHE /speed-scratch/se_gham/hf_cache/metrics
setenv PYTORCH_CUDA_ALLOC_CONF ""




# Start training with debug outputs
echo "Start training at `date`"

setenv ATTN_MODE "xformers"


# usage: sbatch train.sh <train config yml>, e.g.
#   sbatch train.sh options/train/gatedmambair/train_GatedMambaIR_SR_x2_delta_50k.yml
#   sbatch train.sh options/train/gatedmambair/train_GatedMambaIR_SR_x2_gla_50k.yml
#   sbatch train.sh options/train/mambairv2/train_MambaIRv2_SR_x2_50k.yml
if ($#argv < 1) then
  echo "usage: sbatch train.sh <train config yml>"
  exit 1
endif
set CFG = $argv[1]
# one port per job so several runs can share a node
@ PORT = 20000 + ($SLURM_JOB_ID % 10000)
echo "Config: $CFG  master_port: $PORT"

# --auto_resume picks up the latest state in experiments/<name>/training_states if the job is restarted
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT basicsr/train.py -opt $CFG --launcher pytorch --auto_resume

# python /speed-scratch/se_gham/GatedMambaIR/basicsr/archs/chunk_gated_delta_triton.py

echo "done!"
