<pre>source ~/.bashrc
conda activate MODEL_ENV</pre>

go to the data folder of your mamba model
<pre>cd MODEL/data</pre>

paste and save "convert_HarDNet-MSEG.sh" and "convert_HarDNet-MSEG.py" you'll find them in "scripts" folder

just run to download, unzip, convert the dataset and put all the files in the right directory<br>
check pip installations
<pre>bash convert_HarDNet-MSEG.sh</pre>

to augment dataset, put daNUMBER.py that you'll find in the augmentation folder into MODEL/data, then run
<pre>python daNUMBER.py</pre>

default dataset name is Dataset002_HarDNet-MSEG modify if conflict happens

# PREPROCESSING

PREPROCESS DATA on HPC like https://www.dei.unipd.it/bladecluster<br>
using NVIDIA A40 GPU with driver CUDA>12<br>
paste and save file you'll find in the "slurms" folder, specifying the MODEL and MODEL_ENV
<pre>cd ../..
sbatch preproc.slurm</pre>

# TRAINING

paste and save "train_n_evaluate_HarDNet-MSEG.sh" you'll find it in "scripts" folder<br>
paste and save "train.slurm" you'll find in the "slurms" folder, specifying the time, MODEL, MODEL_ENV and MODEL_NAME
<pre>sbatch train.slurm</pre>
