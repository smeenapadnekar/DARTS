<div align="center">

<h1>Distilling Auxiliary RGB-T Features for Unsupervised Semantic Segmentation</h1>

<div>
    <a href='' target='_blank'>S Meena Padnekar</a><sup>1</sup>&emsp;
    <a href='https://www.ee.iitm.ac.in/kmitra/' target='_blank'>Kaushik Mitra</a><sup>1</sup>&emsp;
    <a href='https://www.cse.iitm.ac.in/~sdas/' target='_blank'>Sukhendu Das</a><sup>1</sup>&emsp;
   
   
</div>
<div>
    <sup>1</sup>Visualization & Perception Lab, Department of Computer Science and Engineering, Indian Institute of Technology&emsp; 
    <sup>2</sup>Computational Imaging Lab, Department of Electrical Engineering, Indian Institute of Technology&emsp; 

</div>

<div>
    <strong> 2025</strong>
</div>

<div>
    <h4 align="center">
        • <a href="" target='_blank'>[arXiv]</a> •
    </h4>
</div>

<!--<img src="assets/framework.jpg" width="700px"/> -->

</div>

## Abstract
> *Unsupervised Semantic Segmentation (USS) aims to categorize image pixels into semantic groups without relying on annotated data. While existing USS methods predominantly operate on RGB images and exploit self-supervised Vision Transformers (ViTs) to model semantic correlations, their performance degrades severely under adverse illumination due to the inherent limitations of the RGB modality. To address this challenge, we propose DARTS, a novel multimodal framework that leverages complementary information from the thermal spectrum alongside RGB inputs for unsupervised semantic segmentation.
Observing that self-supervised ViTs produce semantically consistent feature structures across modalities, we design a multimodal feature fusion module equipped with a feature-correlation loss to learn clusterable and illumination-invariant representations from RGB–thermal pairs. The fusion module integrates self- and cross-attention within a single dual-modal ViT block to selectively extract complementary features, followed by a linear fusion mechanism for joint representation learning. To guide the unsupervised training, we introduce intra- and inter-modal feature correlation losses that contrast and distill features within and across modalities, encouraging compact and semantically meaningful pixel embeddings.
DARTS can be seamlessly integrated into existing USS pipelines such as STEGO, SmooSeg, EAGLE, and DepthG, consistently enhancing their segmentation quality under challenging illumination conditions. Extensive experiments on KP, PST900, MFNet, and SemanticRT datasets demonstrate that DARTS achieves superior performance over unimodal baselines, particularly in scenarios with nighttime, glare, or low-visibility environments.*


### Codes and Environment

```
# git clone this repository
git clone https://github.com/smeenapadnekar/DARTS.git
cd DARTS

# create new anaconda env
conda env create -f environment.yml
conda activate darts

```

### Prepare Datasets
Change the `data_dir` variable to your data directory where datasets are stored.

Download datasets and place them in 'data' folder in the following structure:
- [MF dataset](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) or [RTFNet preprocessed version](http://gofile.me/4jm56/CfukComo1)
- [PST900 dataset](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)
- [KP dataset](https://github.com/SoonminHwang/rgbt-ped-detection), [Segmentation label](https://github.com/yeong5366/MS-UDA) or [Pre-organized KP dataset](https://kaistackr-my.sharepoint.com/:u:/g/personal/shinwc159_kaist_ac_kr/EUfmm7hkeaVNuyyYsREttFIBGZ3u_tCmaZ5S5EYghwkKnQ?e=Gyc86F)
- [SemanticRT](https://github.com/jiwei0921/SemanticRT)


python crop_datasets.py
```

### Checkpoints
Download the checkpoints from [[Google Drive](https://drive.google.com/drive/folders/1DueMGFkN6p1RvCxym5BpxsOdm2q3tSCl?usp=drive_link) | [BaiduPan (pw: 2pqh)](https://pan.baidu.com/s/1rK8L7uHmaE5Vun4yLnnL5g?pwd=2pqh)] to `checkpoints` folder.

### Model training
Hyperparameters can be modified in [`SmooSeg/configs/train_config.yaml`](configs/train_config.yaml).
```shell script
CUDA_VISIBLE_DEVICES="0" python train_segmentation.py
```

## Evaluation

To evaluate our pretrained models please run the following in:
```shell script
python src/eval_segmentation.py
```
One can change the evaluation parameters and model by editing [`STEGO/src/configs/eval_config.yml`](src/configs/eval_config.yml)

## Training

To train from scratch, please first generate the KNN indices for the datasets of interest. Note that this requires generating a cropped dataset first, and you may need to modify `crop datasets.py` to specify the dataset that you are cropping:

```shell script
python src/crop_datasets.py
python src/precompute_knns.py
```

Then you can run the following in `DARTS/src`:
```shell script
python src/train_segmentation.py
```
Hyperparameters can be adjusted in [`STEGO/src/configs/train_config.yml`](src/configs/train_config.yml)

## Citation
```
@misc{lan2023smooseg,
      title={SmooSeg: Smoothness Prior for Unsupervised Semantic Segmentation}, 
      author={S Meena Padnekar, Kaushik Mitra and Sukhendu Das},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## Acknowledgement
This study is supported under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s), by the National Research Foundation, Singapore under its Industry Alignment Fund – Pre-positioning (IAF-PP) Funding Initiative, and by the Ministry of Education, Singapore under its MOE Academic Research Fund Tier 2 (STEM RIE2025 Award MOE-T2EP20220-0006).

This implementation is based on [STEGO](https://github.com/mhamilton723/STEGO/tree/master). Thanks for the awesome work.

## Contact
If you have any questions, please feel free to reach out at `lanm0002@e.ntu.edu.sg`.
