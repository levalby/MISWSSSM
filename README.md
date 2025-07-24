# Medical Image Segmentation with Selective State Space Models
bachelor thesis repository<br>

Please cite the following paper when using:<br>
<pre>Alberto Levorato (2025): Medical Image Segmentation with Selective State Space Models<br>
https://hdl.handle.net/****</pre>

                             
Mamba models with nnU-Net structure and Dataset adaptation:<br>
Instruction to work with High Performance Computer like https://www.dei.unipd.it/bladecluster<br>
using NVIDIA A40 GPU with driver CUDA>12<br>
with SLURM scheduler (batch files included)<br>

<pre>miswsssm/
├── Datasets/
│ ├── HarDNet-MSEG/
│ │  └── #instructions to download, convert, and adapt for training
│ ├── Kvasir-SEG/
│ │  └── #instructions to download, convert, and adapt for training
├── Swin-UMamba/
│ └── #instructions to install, setup, train and test on HPC
├── U-Mamba/
│ └── #instructions to install, setup, train and test on HPC
└── README.md</pre>