# CVSSP Audio Captioning System for DCASE challenges 2021 and 2022

## Set up environment

* Clone the repository: `git clone https://github.com/XinhaoMei/DCASE2021_task6_v2.git`
* Create conda environment with dependencies: `conda env create -f environment.yml -n name`
* If you encounter with the `OSError: sndfile library not found `, please try `conda install -c conda-forge libsndfile`
* All of our experiments are running on RTX 3090 with CUDA11. This envirionment just works for RTX 30x GPUs.

## Set up dataset 

* Please refer to `https://github.com/XinhaoMei/audio-text_retrieval`

## Prepare evaluation tool

* Run `coco_caption/get_stanford_models.sh` to download the libraries necessary for evaluating the metrics.

## Run experiments 

### Cross-entropy training

* Set the parameters you want in `settings/settings.yaml` 
* Run experiments: `python train.py -n exp_name`

### Reinforcement learning training

* Set settings in `rl` block in `settings/settings.yaml` 
* Run: `python finetune_rl.py -n exp_name` 

## Cite

For more details, please refer to our technical report [(pdf, 2022)](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Mei_117_t6a.pdf), [(pdf, 2021)](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Mei_88_t6.pdf) and paper [(pdf)](https://arxiv.org/abs/2108.02752).

If you use our code, please kindly cite following:

```
@inproceedings{Mei2021,
    author = "Mei, Xinhao and Huang, Qiushi and Liu, Xubo and Chen, Gengyun and Wu, Jingqian and Wu, Yusong and ZHAO, Jinzheng and Li, Shengchen and Ko, Tom and Tang, H. and Shao, Xi and Plumbley, Mark D. and Wang, Wenwu",
    title = "An Encoder-Decoder Based Audio Captioning System with Transfer and Reinforcement Learning",
    booktitle = "Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)",
    address = "Barcelona, Spain",
    month = "November",
    year = "2021",
    pages = "206--210",
    isbn = "978-84-09-36072-7",
    doi. = "10.5281/zenodo.5770113"
}
```




