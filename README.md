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

For more details, please refer to our technical report [(pdf)](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Mei_88_t6.pdf).

If you use our code, please kindly cite following:

```
@techreport{xinhao2021_t6,
    Author = "Mei, Xinhao and Huang, Qiushi and Liu, Xubo and Chen, Gengyun and Wu, Jingqian and Wu, Yusong and Zhao, Jinzheng and Li, Shengchen and Ko, Tom and Tang, H. Lilian and Shao, Xi and Plumbley, Mark D. and Wang, Wenwu",
    title = "An Encoder-Decoder Based Audio Captioning System With Transfer and Reinforcement Learning for {DCASE} Challenge 2021 Task 6",
    institution = "DCASE2021 Challenge",
    year = "2021",
    month = "July",
    abstract = "Audio captioning aims to use natural language to describe the content of audio data. This technical report presents an automated audio captioning system submitted to Task 6 of the DCASE 2021 challenge. The proposed system is based on an encoder-decoder architecture, consisting of a convolutional neural network (CNN) encoder and a Transformer decoder. We further improve the system with two techniques, namely, pre-training the model via transfer learning techniques, either on upstream audio-related tasks or large in-domain datasets, and incorporating evaluation metrics into the optimization of the model with reinforcement learning techniques, which help address the problem caused by the mismatch between the evaluation metrics and the loss function. The results show that both techniques can further improve the performance of the captioning system. The overall system achieves a SPIDEr score of 0.277 on the Clotho evaluation set, which outperforms the top-ranked system from the DCASE 2020 challenge."
}
```



