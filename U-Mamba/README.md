Instructions to install, setup and train U-Mamba on HPC like https://www.dei.unipd.it/bladecluster<br>
using NVIDIA A40 GPU with driver CUDA>12<br>
slurm files in the end of the document, copy and paste them when needed<br>
use vim to work with files: [i] to insert, [ESC] to stop, [:wq] to save and quit<br>
if you encounter any error with preprocessing and training, make sure everything is working correctly, go to the end of this file and follow "Sanity check" steps<br>

# INSTALLATION

install miniconda to set up a virtual environment in your partition
<pre>wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh</pre>

request an interactive job to use A40 GPU
<pre>srun --pty --mem=1g --time=01:00:00 -n 1 --gres=gpu:a40:1 -J interactive -p interactive /bin/bash
</pre>

setup conda environment
<pre>source ~/.bashrc
conda create -n umamba python=3.10 -y
conda activate umamba
pip install torch==2.5.1 torchvision==0.20.1
pip install causal-conv1d>=1.2.0
pip install mamba-ssm --no-cache-dir
git clone https://github.com/bowang-lab/U-Mamba
cd U-Mamba/umamba
pip install -e .</pre>


this version is required, numpy<1.20 && numpy>2 are not supported for the model
<pre>pip install numpy==1.26.4</pre>


Sanity test
<pre>python -c "import torch; import mamba_ssm"
# if output does not correspond to expected, maybe your installation is broken, pip uninstall and reinstall it
python -c "import numpy; print(numpy.__version__)"
# 1.26.4
python -c "import torch; print(torch.__version__)"
# 2.5.1
python -c "import torchvision; print(torchvision.__version__)"
# 0.20.1
python -c "import torch; print(torch.rand(5, 3))"
# tensor([[0.3380, 0.3845, 0.3217], ...])
python -c "import torch; print(torch.cuda.is_available())"
# True
CTRL+D to exit the interactive env</pre>

# DOWNLOAD DATA
<pre>cd ../data
#useful for dataset downloading
pip install gdown</pre>
choose one or more dataset you want to work with

Download Dataset702_AbdomenMR
<pre>wget "https://drive.usercontent.google.com/download?id=1MIk6dybEyMTjkdmljw5_Ydj6u56L5lbY&export=download&confirm=t&uuid=7451050d-1cf5-4d5b-a464-08694b97bf72" -O "Dataset702_AbdomenMR.zip" -nc
rm -rf "nnUNet_raw/Dataset702_AbdomenMR"
unzip -q "Dataset702_AbdomenMR.zip" -d "nnUNet_raw"</pre>

Download Dataset703_NeurIPSCell
<pre>gdown --fuzzy "https://drive.google.com/drive/folders/1N2DfyT0uweyvm7Nt-OmEvb9QFX6HsdF3?usp=drive_link"
rm -rf "nnUNet_raw/Dataset703_NeurIPSCell"
unzip -q "Dataset703_NeurIPSCell.zip" -d "nnUNet_raw"</pre>

Download Dataset704_Endovis17 
<pre>gdown --fuzzy "https://drive.google.com/drive/folders/1FPyZjsGC8fzWWAkqTWH8Imy2O-k1yKIB?usp=drive_link"
rm -rf "nnUNet_raw/Dataset704_Endovis17"
unzip -q "Dataset704_Endovis17.zip" -d "nnUNet_raw"</pre>

# PREPROCESS DATA

<pre>cd ../..</pre>

copy the file you'll find in the "slurms" folder, specifying: DATASET_ID<br>
specify DATASET_ID, for AbdomenMR put 702<br>
<pre>sbatch preproc.slurm</pre>

# TRAIN MODEL

prepare scripts
<pre>cd U-Mamba
mkdir scripts
cd scripts</pre>

copy here the desired script, BASH files to put in U-Mamba/scripts you'll find in "scripts" folder<br>
- train_AbdomenCT.sh<br>
- train_AbdomenMR.sh<br>
- train_Endoscopy.sh<br>
- train_Microscopy.sh<br>

<pre>cd ../..</pre>

copy the file you'll find in the "slurms" folder, specifying: SCRIPT_NAME, MODEL_NAME, hh:mm:ss and adjust the GPU-memory configuration<br>
- SCRIPT_NAME, for AbdomenMR put AbdomenMR<br>
- MODEL_NAME: nnUNetTrainerUMambaBot or nnUNetTrainerUMambaEnc<br>
- DIMENSION: 2d or 3d_fullres<br>
- NUM_GPUS: 1 or 2<br>
- specify the time in -t, to run 100 epochs with 1 A40 GPU approximately 15h were needed, memory usage for batch_size=30 is approximately 25G<br>
to edit the batch go to /U-Mamba/data/nnUNet_preprocessed/DatasetDATASET_ID_SCRIPT_NAME/nnUNetPlans.json in "configurations"/"2d"/"batch_size"<br>
<pre>sbatch train.slurm</pre>

to check how job is performing, specify JOB_ID
<pre>srun --pty --jobid JOB_ID /bin/bash
nvtop</pre>

Error you may encounter in a limited disk partition on cluster: "OSError: [Errno 122] Disk quota exceeded"<br>
Check the hidden folder ".conda" and ".cache" sizes, they may add up to several GBs after some work, delete them and start your train over<br>
Delete unused checkpoints