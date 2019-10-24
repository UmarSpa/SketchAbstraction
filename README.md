# Sketch abstraction from "Goal-Driven Sequential Data Abstraction"

<p align="center">
<img src="https://umarriaz.org/wp-content/uploads/2019/10/Preview-ICCV19.png" width="500">
</p>

## Introduction

This repo is official Tensorflow implementation of [Goal-Driven Sequential Data Abstraction (ICCV 2019)](https://arxiv.org/pdf/1907.12336.pdf). It contains sketch abstraction part.

## Repo structure
```
${SkechAbstraction}
|-- Source
|-- |-- InData
|   |-- Classifier
|   |   |-- Weights
|   |-- Agent
|   |   |-- Weights
|-- Output
```
* Download processed sketch dataset - 9 classes from QuickDraw [[InData](https://drive.google.com/file/d/1zEQTM3a8a9EOXXdpgRl9hB6574YgC3Xm/view?usp=sharing)]
* Download pretrained classifier weights [[Weights](https://drive.google.com/drive/folders/1qULg2XieNYa_aI4pyK5YxX_4WPdHlNCc?usp=sharing)]
* Download pretrained agent weights [[Weights](https://drive.google.com/drive/folders/16MIflRh_iDrFKRbgVH19VzFic2-YXN9q?usp=sharing)]

## Dependencies
The requirements.txt file lists all the required dependencies, which can be installed using:
```bash
pip install -r requirements.txt
```

## Example usage
Train with CNN classifier and 25% budget size:
```bash
python main.py --trainFlag --classType 'CNN' --budgetVal 0.25 --outDir './Output/RNN-25/'
```
Test with CNN classifier and 25% budget size:
```bash
python main.py --classType 'CNN' --budgetVal 0.25 --agentLoading --agentLoadingFile './Source/Agent/Weights/Ag
ent_Weights_25_CNN.npy' --testStep 1
```

N.B. For training and testing of models with RNN classifier and budget size of 50%, please change the `--classType` and `--budgetVal` values accordingly.

## Results
 Category recognition (acc. %) of the abstracted sketches.
 
|             | RNN (25%) | RNN (50%) | CNN (25%) | CNN (50%) |
|-------------|:---------:|:---------:|:---------:|:---------:|
| Human       |   36.66   |   66.73   |   62.08   |   75.90   |
| Random      |   22.67   |   45.65   |   41.06   |   65.47   |
| DSA         |   38.36   |   67.89   |   65.05   |   81.50   |
| DQSN        |   38.11   |   67.50   |   64.58   |   80.31   |
| GDSA (Ours) |   50.50   |   71.75   |   71.92   |   86.15   |

## Reference
```
@article{Muhammad_2019_ICCV_GDSA,
  author = {Umar Riaz Muhammad and Yongxin Yang and Timothy M. Hospedales and Tao Xiang and Yi-Zhe Song},
  title = {Goal-Driven Sequential Data Abstraction},
  booktitle = {The IEEE Conference on International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```
