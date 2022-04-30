# 294-082-final-project

Evaluation the Experimental Design of Land Cover Classification Model


## Quick start

### 1. Setup Envionment

1. Create a new conda environment and activate it
    ```
    conda create --name cs294
    conda activate cs294
    conda install pip
    ```
2. Install requirement packages
    ```
    pip install -r requirements.txt
    ```
    You might want to install the cuda version of pytorch again using the cmd [here](https://pytorch.org/get-started/locally/) if you want to use gpu for pytorch

### 2. Run 1 GPU baseline
1. run on your own laptop
    ```
    python -m model_baseline.resnet-eurosat
    ```
1. run on bridge2
    ```
    # login in to interactive node
    salloc -N 1 -n 1 -p GPU-shared --gres=gpu:1 -q interactive -t 01:00:00

    # load pytorch environment
    singularity shell  /ocean/containers/ngc/pytorch/pytorch_latest.sif

    # run code
    python -m model_baseline.resnet-eurosat
    ```
1. run on bridges-2 job
    
    change fc_nodes in bridge_job.sh then run
    ```
    sbatch bridge_job.sh
    ```

### 3. Total workflow

1. Download the dataset and try start resnet base training
    ```
    python -m model_baseline.resnet-eurosat
    ```
1. Generate csv data from RGB photo and resnet output
    ```
    python -m scripts.image-to-csv
    python -m scripts.resnet-to-csv
    ```
1. run brainome
    ```
    python -m brainome login
    python -m brainome -vv -target class -o resnet-nn.py -f NN  ./data/resnet-output.csv
    ```
1.  Training model with different hiden layer nodes
    ```sh
    for i in {1..15}
    do
        python -m model_baseline.mlp-eurosat --fc_nodes $i
    done
    ```

1. run notebook under ./scripts/data_mec.ipynb to get dataset mec, Capacity Progression and plots

### 4. Pretrained Model and Logs

