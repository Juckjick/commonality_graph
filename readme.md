# Code Appendix for Evaluation

Leveraging Commonality across Multiple Tissue Slices for Enhanced Whole Slide Image Classification Using Graph Convolutional Networks
Ref: Submission ID 37c9a612-3247-4768-b8d0-6e71519d2c5f

## Pre-requisites

- Python
- Pytorch
- Pytorch Geometric
- torchmetrics
- NetworkX
- Numpy
- Pandas
- Pillow; PIL

## Data Preparation

First, we tile a whole slide image (WSI) into many patches with a size of 256*256 pixels, and remove background (white) and noisy patches. Also, we generate a Comma Separated Value (CSV) file to list patch information described in a following table.

| columns | descriptions | value |
| --- | --- | --- |
| organ | an organ type of a patch | ‘colon’ or ‘stomach’ |
| subset | a train-val-test split type | ‘train’ or  ‘val’ or ‘test’ |
| condition | a slide class | 'D’ or ‘M’ or ‘N’ |
| label | a slide label mapping from the condition column:
0, 1, and 2 are for D, M, and N, respectively. | 0 or 1 or 2 |
| slide_name | a name of a slide | string |
| img_path | a path to patch image. For more detail, patches are named in this form: “…/slide_name-row_column.jpg” | string |

Samples of patch information (CSV file) are as follows:

| organ | subset | condition | label | slide_name | img_path |
| --- | --- | --- | --- | --- | --- |
| colon | train | D | 0 | 2018S016592203 | ../2018S016592203-100_62.jpg |
| colon | train | D | 0 | 2018S016592203 | ../2018S016592203-100_63.jpg |
| stomach | val | M | 1 | 2021S 0067167010104 | ../2021S 0067167010104-52_59.jpg |
| stomach | val | M | 1 | 2021S 0067167010104 | ../2021S 0067167010104-53_31.jpg |

## Feature Extraction

Patch features are extracted in this step. An input data is the patch information (csv file) from previous step. And, the patch features are the output as a pickle (pkl) file. Due to the restriction of sharing the raw dataset, we cannot provide the tile information. However, we provide the output of this step in `files/features`. Although you cannot run this step, we offer the code (`feature_extractor.py`) for a supplementary. Noted that a denoising algorithm from [this paper](https://www.nature.com/articles/s41598-022-05001-8) is used to train the feature extractor.

## Graph Construction

Then, we construct a commonality graph by leveraging common patterns across slices from their extracted features. The outputs are commonality graphs stored in `files/graphs`. 

To run this step, you can follow this command:

```jsx
$ python graph_extractor.py
```

## Training and testing the model

After generating graphs from previous step, you can run the experiment with our proposed data representative. The code for training and testing graph convolutional neural networks (GCNs) are provided. The input graphs are from `files/graphs`. The outputs of this step are the trained models and their performances in terms of accuracy and AUROC. In addition, the trained models are stored in `models/trained`.

To run this step, you can follow this command:

```jsx
$ python main_commonality.py
```
