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

## Run experiments 

### Cross-entropy training

* Set the parameters you want in `settings/settings.yaml` 
* Run experiments: `python train.py -n exp_name`

### Reinforcement learning training

* Set settings in `rl` block in `settings/settings.yaml` 
* Run: `python finetune_rl.py -n exp_name` 

## Reproduce results 

* Four pre-trained models submitted to DCASE 2021 Task 6 are under `pretrained_model/models/submission{1-4}`
* Change `mode` in `settings\settings.yaml` to `"eval"` and model path to the path of these pre-trained models, you can get the results displayed in our technical report

## Cite

For more details, please refer to our technical report [(pdf)](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Mei_88_t6.pdf) and paper [(pdf)](https://arxiv.org/abs/2108.02752).

If you use our code, please kindly cite following:

```
@techreport{xinhao2021_t6,
    Author = "Mei, Xinhao and Huang, Qiushi and Liu, Xubo and Chen, Gengyun and Wu, Jingqian and Wu, Yusong and Zhao, Jinzheng and Li, Shengchen and Ko, Tom and Tang, H. Lilian and Shao, Xi and Plumbley, Mark D. and Wang, Wenwu",
    title = "An Encoder-Decoder Based Audio Captioning System With Transfer and Reinforcement Learning for {DCASE} Challenge 2021 Task 6",
    institution = "DCASE2021 Challenge",
    year = "2021",
    month = "July",
}
```

or:

```
@article{mei2021encoder,
  title={An Encoder-Decoder Based Audio Captioning System With Transfer and Reinforcement Learning},
  author={Mei, Xinhao and Huang, Qiushi and Liu, Xubo and Chen, Gengyun and Wu, Jingqian and Wu, Yusong and Zhao, Jinzheng and Li, Shengchen and Ko, Tom and Tang, H Lilian and others},
  journal={arXiv preprint arXiv:2108.02752},
  year={2021}
}
```



