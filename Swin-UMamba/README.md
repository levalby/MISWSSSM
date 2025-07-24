Instructions to install, setup and train Swin-Umamba on HPC like https://www.dei.unipd.it/bladecluster<br>
using NVIDIA A40 GPU with driver CUDA>12<br>
slurm files in the end of the document, copy and paste them when needed<br>
use vim to work with files: [i] to insert, [ESC] to stop, [:wq] to save and quit<br>

# INSTALLATION

install miniconda to set up a virtual environment in your partition
<pre>wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh</pre>
    
request an interactive job to use A40 GPU
<pre>srun --pty --mem=1g --time=01:00:00 -n 1 --gres=gpu:a40:1 -J interactive -p interactive /bin/bash</pre>

setup conda environment
<pre>source ~/.bashrc
conda create -n swin_umamba python=3.10 -y
conda activate swin_umamba</pre>

install dependencies for the model, torch and torchvision versions are required to work with CUDA>12
<pre>pip install torch==2.5.1 torchvision==0.20.1
pip install causal-conv1d==1.1.1 # causal-conv1d>=1.2.0
pip install mamba-ssm
pip install torchinfo timm numba
git clone https://github.com/JiarunLiu/Swin-UMamba
cd Swin-UMamba/swin_umamba
pip install -e .</pre>


this version is required, numpy<1.20 && numpy>2 are not supported for the model
<pre>pip install numpy==1.26.4</pre>

edit the training script, set the established number of GPUs you want to use to train the Dataset (default is 2)
<pre>cd ../scripts
vim train_DATASET_NAME.sh</pre>
EXAMPLE: train_AbdomenMR.sh, line 9, set: --num_gpus 1


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

choose one or more dataset you want to work with<br>

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
wget "https://drive.usercontent.google.com/download?id=&export=download&confirm=t&uuid=GETFULLLINK" -O "Dataset704_Endovis17.zip" -nc
rm -rf "nnUNet_raw/Dataset704_Endovis17"
unzip -q "Dataset704_Endovis17.zip" -d "nnUNet_raw"</pre>

# PREPROCESS DATA

<pre>cd ../..</pre>


paste and save file you'll find in "slurms" folder specifying: DATASET_ID<br>
specify DATASET_ID, for AbdomenMR put 702
<pre>sbatch preproc.slurm</pre>

# PREPARE PRETRAINED MODEL
<pre>wget https://github.com/MzeroMiko/VMamba/releases/download/%23v0cls/vssmtiny_dp01_ckpt_epoch_292.pth
mv vssmtiny_dp01_ckpt_epoch_292.pth Swin-UMamba/data/pretrained/vmamba/vmamba_tiny_e292.pth</pre>



# TRAIN MODEL


paste and save the file you'll find in "slurms" folder, specifying: SCRIPT_NAME, MODEL_NAME, hh:mm:ss and adjust the GPU-memory configuration<br>
specify SCRIPT_NAME, for AbdomenMR put AbdomenMR<br>
specify MODEL_NAME, to train AbdomenMR with vmamba_tiny_e292.pth was used: nnUNetTrainerSwinUMamba, can be:<br>
- nnUNetTrainerSwinUMamba: Swin-UMamba model with ImageNet pretraining<br>
- nnUNetTrainerSwinUMambaD: Swin-UMamba$\dagger$ model with ImageNet pretraining<br>
- nnUNetTrainerSwinUMambaScratch: Swin-UMamba model without ImageNet pretraining<br>
- nnUNetTrainerSwinUMambaDScratch: Swin-UMamba$\dagger$ model without ImageNet pretraining<br>
specify the time in -t, to run 100 epochs with 1 A40 GPU approximately 15h were needed, memory usage for batch_size=30 is approximately 25G<br>
to edit the batch go to /Swin-UMamba/data/nnUNet_preprocessed/DatasetDATASET_ID_SCRIPT_NAME/nnUNetPlans.json in "configurations"/"2d"/"batch_size"<br>
<pre>sbatch train.slurm</pre>


to check how job is performing, specify JOB_ID
<pre>srun --pty --jobid JOB_ID /bin/bash
nvtop</pre>

Error you may encounter in a limited disk partition on cluster: "OSError: [Errno 122] Disk quota exceeded"<br>
Check the hidden folder ".conda" and ".cache" sizes, they may add up to several GBs after some work, delete them and start your train over<br>
Delete unused checkpoints