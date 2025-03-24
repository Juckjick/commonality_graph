# 3D Graph-Based Classification of  Histopathological Whole Slide Images (#12754) - Code Appendix

## Pre-requisites

- Python (3.6.15)
- Pytorch (1.8.0=py3.6_cuda10.2_cudnn7_0)
- Pytorch Geometric (1.7.0)
- torchmetrics (0.9.3)
- NetworkX (2.5.1)
- Numpy (1.19.5)
- Pandas (1.1.5)
- Pillow; PIL (8.3.2)
- tqdm (4.62.3)

## Data Preparation

First, we cut a whole slide image (WSI) into tiles with a size of 256*256 pixels, and then adapt a denoising algorithm from [this paper](https://www.nature.com/articles/s41598-022-05001-8) to get rid of background and noisy patches. Furthermore, we generate a Comma Separated Value (CSV) file to list tile information, which is described in the following table.

| columns | descriptions | value |
| --- | --- | --- |
| organ | an organ type of a tile | ‘colon’ or ‘stomach’ |
| subset | a train-val-test split type | ‘train’ or  ‘val’ or ‘test’ |
| condition | a slide class | 'D’ or ‘M’ or ‘N’ |
| label | a slide label mapping from the condition column:
0, 1, and 2 are for D, M, and N, respectively. | 0 or 1 or 2 |
| slide_name | a name of a slide | string |
| img_path | a path to tile image. For more detail, tiles are named in this form: “…/slide_name-row_column.jpg” | string |

The samples of tile information (CSV file) are as follows:

| organ | subset | condition | label | slide_name | img_path |
| --- | --- | --- | --- | --- | --- |
| colon | train | D | 0 | 2018S016592203 | ../2018S016592203-100_62.jpg |
| colon | train | D | 0 | 2018S016592203 | ../2018S016592203-100_63.jpg |
| stomach | val | M | 1 | 2021S 0067167010104 | ../2021S 0067167010104-52_59.jpg |
| stomach | val | M | 1 | 2021S 0067167010104 | ../2021S 0067167010104-53_31.jpg |

## Feature Extraction

The tile’s features are extracted. The input data is the CSV file of tile information from the previous step. And, the tile’s features are the output as a pickle (pkl) file. Due to the restriction of sharing the raw dataset, we cannot provide the tile information. However, we provide the output of this step in `files/features`. Although you cannot run this step, we offer the code (`feature_extracr.py`) for a supplementary. 

## Graph Construction

We construct a three-dimensional (3D) graph using the features extracted from the previous step. The outputs are 3D graphs stored in `files/graphs`. 

To run this step, you can follow this command:

```jsx
$ python graph_extractor.py
```

## Training and testing the model

After generating graphs from the previous step, you can run the experiment with our proposed data representative. The code for training and testing graph convolutional neural networks (GCNs) are provided. The input graphs are from `files/graphs`. The outputs of this step are the trained models and their performances in terms of accuracy and AUROC. In addition, the trained models are stored in `models/trained`.

To run this step, you can follow this command:

```jsx
$ python main_3DGCN.py
```