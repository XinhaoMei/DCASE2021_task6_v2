# AN ENCODER-DECODER BASED AUDIO CAPTIONING SYSTEM WITH TRANSFER AND REINFORCEMENT LEARNING FOR DCASE CHALLENGE 2021 TASK 6

## Set up environment

* Clone the repository: `git clone https://github.com/XinhaoMei/DCASE2021_task6_v2.git`
* Create conda environment with dependencies: `conda create -f environment.yml -n name`
* If you encounter with the `OSError: sndfile library not found `, please try `conda install -c conda-forge libsndfile`
* All of our experiments are running on RTX 3090 with CUDA11. This envirionment just works for RTX 30x GPUs.

## Set up dataset 

* Run download_dataset.sh to download the dataset: `./download_dataset.sh`
* The file of vocabulary has been placed under `data/pickles`
*  Create dataset: `python dataset_creation.py`

## Prepare evaluation tool

* Run `coco_caption/get_stanford_models.sh` to download the libraries necessary for evaluating the metrics.

## Run experiment 

* Set the parameters you want in `settings/settings.yaml`
* Run experiments: `python train.py -n exp_name`

## Reproduce results 

* Four pre-trained models submitted to DCASE 2021 Task 6 are under `pretrained_model/models/submission{1-4}`
* Change `mode` in `settings\settings.yaml` to `"eval"` and model path to the path of these pre-trained models, you can get the results displayed in our technical report

## Cite





