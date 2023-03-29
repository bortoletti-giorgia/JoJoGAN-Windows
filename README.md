# Windows Installation Tutorial
**(The JoJoGAN code will not be updated)**

## Run JoJoGAN locally

Small fixes compared to [JoJoGAN-Training-Windows](https://github.com/bycloudai/JoJoGAN-Training-Windows).

Using [JoJoGAN-Training-Windows](https://github.com/bycloudai/JoJoGAN-Training-Windows) guide on NVIDIA RTX 2060 after running 
```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```, it was installed  ```pytorch from pytorch/win-64::pytorch-1.13.1-py3.7_cpu_0```.
In this way, CUDA was not found and Pytorch was executed with CPU so I found the combination of Python packages versions which is in the next section.

#### Step 0
On Windows 11 install:
* [Anaconda3 2022](https://www.anaconda.com/products/distribution)
* [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
* Visual Studio Community 2019
* NVIDIA RTX 2060

Environment Variables should be:
```
CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
Path = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin
      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\libnvvp
      C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat
      C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
      C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin
```

### Step 1
Create a new Anaconda environment.

```sh
conda create -n jojo python=3.7
conda activate jojo
```

Otherwise, download the Anaconda [environment](https://github.com/bortoletti-giorgia/JoJoGAN-Windows/blob/main/extra/environment.yml) and go directly to the Final Step.

```sh
conda env create -f environment.yml
conda activate jojo
```

### Step 2
Install Pytorch related.
```sh
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

If install Pytorch with Pip as:
```sh
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
you could have this error:
```sh
import dlib
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: DLL load failed: The specified module could not be found.
```

### Step 3
Install Python requirement packages.
```sh
pip install tqdm gdown scikit-learn==0.22 scipy lpips dlib==19.20 opencv-python wandb matplotlib scikit-image pybind11 cmake ninja
conda install -c conda-forge ffmpeg
```

### Step 4
Clone the JoJoGAN project.
```sh
git clone https://github.com/mchong6/JoJoGAN.git
cd JoJoGAN
```

### Final Test 

```sh
(jojo) C:\JoJoGAN>python
Python 3.7.16 (default, Jan 17 2023, 16:06:28) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import dlib
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```

Import some JoJoGAN specific packages without errors.
```sh
(jojo) C:\JoJoGAN>python
Python 3.7.16 (default, Jan 17 2023, 16:06:28) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> from model import *
>>> from e4e_projection import projection as e4e_projection
>>> from util import *
>>>
```

If there is an error than contains ```torch\utils\cpp_extension.py:237: userwarning: error checking compiler version for cl```, make sure that the Path "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" includes a "cl.exe" file otherwise search in the folder "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin" for the right one.



### Run JoJoGAN

Activate jojo env and launch jupyter-notebook.
Clone this repository and copy [run-local.py](https://github.com/bortoletti-giorgia/JoJoGAN-Windows/blob/main/extra/run-local.ipynb) created from [https://github.com/prodramp/DeepWorks/tree/main/JoJoGAN](https://github.com/prodramp/DeepWorks/tree/main/JoJoGAN) in the main folder.
Run the code step-by-step.


## Run JoJoGAN in Cluster DEI of University of Padua

If you don’t have at least of 12 GB in your GPU and it’s not RTX 3090 or Tesla V100, you can run the code in [SLURM CLUSTER DEI](https://clusterdeiguide.readthedocs.io/en/latest/index.html).

Requirements to access SLURM:
* Windows 11
* An account DEI: ask for it here https://www.dei.unipd.it/helpdesk/index.php
* At least 20 GB in your workspace
* WinSCP with PuTTY

Requirements to create the Singularity Container:
* Ubuntu
* Singularity

### Create Singularity Container

For running your code in SLURM, you need to create a Singularity Container.
I created the container in Ubuntu because it requires fewer applications and has fewer conflicts than Windows but Singularity can also be installed on Windows and produces the same results.

Create the container from the Singularity Definition file [singularity-container.def](https://github.com/bortoletti-giorgia/JoJoGAN-Windows/blob/main/extra/singularity-container.def).

Open Command Prompt and write: ```sudo singularity build singularity-container.sif singularity-container.def```

If you want to modify something you can run: ```singularity shell singularity-container.sif```.

The singularity-container.sif container contains:
* Ubuntu 18.04
* CUDA 11.1 with its location saved in the PATH
* Ninja package
* Anaconda 2020
* An environment Anaconda called *jojo* with:
    * Python 3.7
    * ```pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html ```
    * ```pip install cmake```
    * ```pip install dlib==19.20 tqdm gdown scikit-learn==0.22 scipy lpips opencv-python wandb mat-plotlib scikit-image pybind11 ninja```
    * ```conda install -c conda-forge ffmpeg```

### Run on Cluster DEI

Login to Windows 11 and download [main.job](https://github.com/bortoletti-giorgia/JoJoGAN-Windows/blob/main/extra/run-remote.job). Be careful to rename it as *main.job*.
Download also [main.py](https://github.com/bortoletti-giorgia/JoJoGAN-Windows/blob/main/extra/main.py) if you want to run JoJoGAN with pretrained model or [main-create-own-style.py](https://github.com/bortoletti-giorgia/JoJoGAN-Windows/blob/main/extra/main-create-own-style.py) if you want to create a model with your style references. Both of the latter codes require arguments to their invocation. Check *main.job* to see if they are present with the ones you want.

Open WinSCP and connect to *login.dei.unipd.it* using SCP protocol.

Your workspace structure should be (“bortoletti” is the example workspace):

```
    \home\bortoletti
    ├── JoJoGAN                           # clone of https://github.com/mchong6/JoJoGAN 
    ├── ├── inversion_codes               # folder created after execution of main.py
    ├── ├── style_images                  # folder created after execution of main.py
    ├── ├── style_images_aligned          # folder created after execution of main.py
    ├── ├── models                        # folder created after execution of main.py
    ├── ├── results                       # folder created after execution of main.py
    ├── ├──  main.py                      # main code to run JoJoGAN with pretrained model
    ├── ├──  main-create-own-style.py     # main code to create a model with your style images 

    ├── out                               # folder with TXT file with errors and shell output of main.job 
    │   main.job                          # JOB file for running JoJoGAN 
    │   singularity-container.sif         # Singularity container for executing the job file
```


Open PuTTY and write: ```sbatch main.job```. Now 
At the end you can find in folder:
* *./out*: one TXT file with a list of errors and one TXT file with output of the job;
*	*/JoJoGAN/results*: images resulted from main.py.




# JoJoGAN: One Shot Face Stylization
[![arXiv](https://img.shields.io/badge/arXiv-2112.11641-b31b1b.svg)](https://arxiv.org/abs/2112.11641)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mchong6/JoJoGAN/blob/main/stylize.ipynb)
[![Replicate](https://replicate.com/mchong6/jojogan/badge)](https://replicate.com/mchong6/jojogan)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/JoJoGAN)
[![Wandb Report](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)](https://wandb.ai/akhaliq/jojogan/reports/JoJoGAN-One-Shot-Face-Stylization-with-Wandb-and-Gradio---VmlldzoxNDMzNzgx)


![](teasers/teaser.jpg)

This is the PyTorch implementation of [JoJoGAN: One Shot Face Stylization](https://arxiv.org/abs/2112.11641).


>**Abstract:**<br>
While there have been recent advances in few-shot image stylization, these methods fail to capture stylistic details
that are obvious to humans. Details such as the shape of the eyes, the boldness of the lines, are especially difficult
for a model to learn, especially so under a limited data setting. In this work, we aim to perform one-shot image stylization that gets the details right. Given
a reference style image, we approximate paired real data using GAN inversion and finetune a pretrained StyleGAN using
that approximate paired data. We then encourage the StyleGAN to generalize so that the learned style can be applied
to all other images.

## Updates

* `2021-12-22` Integrated into [Replicate](https://replicate.com) using [cog](https://github.com/replicate/cog). Try it out [![Replicate](https://replicate.com/mchong6/jojogan/badge)](https://replicate.com/mchong6/jojogan)

* `2022-02-03` Updated the paper. Improved stylization quality using discriminator perceptual loss. Added sketch model
<br><img src="teasers/sketch.gif" width="50%" height="50%"/>
* `2021-12-26` Added wandb logging. Fixed finetuning bug which begins finetuning from previously loaded checkpoint instead of the base face model. Added art model <details><br><img src="teasers/art.gif" width="50%" height="50%"/></details>

* `2021-12-25` Added arcane_multi model which is trained on 4 arcane faces instead of 1 (if anyone has more clean data, let me know!). Better preserves features <details><img src="teasers/arcane.gif" width="50%" height="50%"/></details>

* `2021-12-23` Paper is uploaded to [arxiv](https://arxiv.org/abs/2112.11641).
* `2021-12-22` Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try it out [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/JoJoGAN)
* `2021-12-22` Added pydrive authentication to avoid download limits from gdrive! Fixed running on cpu on colab.



## How to use
Everything to get started is in the [colab notebook](https://colab.research.google.com/github/mchong6/JoJoGAN/blob/main/stylize.ipynb).

## Citation
If you use this code or ideas from our paper, please cite our paper:
```
@article{chong2021jojogan,
  title={JoJoGAN: One Shot Face Stylization},
  author={Chong, Min Jin and Forsyth, David},
  journal={arXiv preprint arXiv:2112.11641},
  year={2021}
}
```

## Acknowledgments
This code borrows from [StyleGAN2 by rosalinity](https://github.com/rosinality/stylegan2-pytorch), [e4e](https://github.com/omertov/encoder4editing). Some snippets of colab code from [StyleGAN-NADA](https://github.com/rinongal/StyleGAN-nada)
