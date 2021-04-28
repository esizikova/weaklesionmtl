# Improving Weakly Supervised Lesion Segmentation using Multi-Task Learning
Tianshu Chu* (New York University), Xinmeng Li* (New York University), Huy V. Vo (Ecole Normale Superieure, INRIA and Valeo.ai), Ronald M. Summers (National Institutes of Health Clinical Center), Elena Sizikova (New York University)
*- equal contribution.

## Introduction
This is a code release of the paper "Weakly Supervised Lesion Segmentation using Multi-Task Learning". 

We introduce the concept of multi-task learning to weakly-supervised lesion segmentation, one of the most critical and challenging tasks in medical imaging. Due to the lesions' heterogeneous nature, it is difficult for machine learning models to capture the corresponding variability. We propose to jointly train a lesion segmentation model and a lesion classifier in a multi-task learning fashion, where the supervision of the latter is obtained by clustering the RECIST measurements of the lesions.

<div align=center>
<img src="images/overview.png" width="600"/>
</div>

### License
This work is released under the GNU General Public License (GNU GPL) license.

## Requirements
----
1. python3.8 was tested
2. pytorch 1.7.1 version was tested 
3. Please refer to the requirements.txt file for other dependencies.

## Installation
1. pip install -r requirements.txt
2. git clone https://github.com/qubvel/segmentation_models.pytorch.git

## HAM10K Dataset
### Download and unzip to the folder code_ham/Data_ham:

1. Raw images: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Ground true segmentations: https://www.kaggle.com/tschandl/ham10000-lesion-segmentations?select=HAM10000_segmentations_lesion_tschandl
3. For data preprocessing:
```
python code_ham/HAM_data.py
```

### Training
Under code_ham/ directory

1. To train A1 model in the paper: 
```
python HAM_A1_train.py
```
2. To train A1+L model in the paper:
```
python HAM_A1class_train.py
```
3. To train Acoseg model in the paper:
```
python HAM_Acoseg_train.py
```

### Evaluation
Under code_ham/ directory

1. To evaluate A1 model in the paper: 
```
python HAM_A1_eval.py
```
2. To evaluate A1+L model in the paper:
```
python HAM_A1class_eval.py
```
3. To evaluate Acoseg model in the paper:
```
python HAM_Acoseg_eval.py
```

### Trained checkpoints
Put under the folder code_ham/checkpoints to evaluate:  

1. A1: https://drive.google.com/file/d/12gK0K_SZNleFAFVwPjzspHIS3rsWGXEb/view?usp=sharing
2. A1+L: https://drive.google.com/file/d/1yFIfjXay9TRw_Wn_QLbfAdhMpFhcAqdC/view?usp=sharing
3. ACoseg: https://drive.google.com/file/d/1B-3bG26yqpnupkH-q-qNHp1_IrU4EQf4/view?usp=sharing


## LiTS and DeepLesion Dataset
TODO 

## Results
----
Cluster visualizations can be seen below. Clustering based on RECIST parameters allows grouping lesions into groups with similar shapes.
<div align=center>
<img src="images/clusters.png" width="600"/>
</div>

Visualization of sample segmentation results. 1st row - input lesion image. 2nd row - ground truth. 3rd row - appeoximate target. 4th row - result of A1 baseline, based on DeepLabV3+ (A1). 5th row - result of co-segmentation baseline (ACoseg), [Agarwal, 2020a]. 6th row - our result (A1+L). 
<div align=center>
<img src="images/segmentation_results.png" width="600"/>
</div>

## Citation

```
@inproceedings{chuli2021lesion_wsol,
	title={Improving Weakly Supervised Lesion Segmentation Using Multi-Task Learning},
	author={Chu, Tianshu and Li, Xinmeng and Vo, Huy V. and Summers, Ronald M. and Sizikova, Elena},
	booktitle={Proceedings of the Medical Imaging with Deep Learning (MIDL) Conference},
	year={2021}
}
```

