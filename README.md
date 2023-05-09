# <p align=center>`RGB-D Salient Object Detection datasets`</p>

This repo is responsible for managing the **datasets** related to **RGB-D SOD**

- Push the dataset to [HuggingFace](https://huggingface.co/datasets)
- Conversion between mask and polygons (compatible to [Labelme](https://github.com/wkentaro/labelme) format)

## Conversion between mask and polygons

- See `mask.py` for more information
- We can convert from mask to polygons and vice versa.

<img src="images/mask_to_polygons.png"/>

## Prepare datasets directory structure

```
├── rgbdsod_datasets
│   ├── train
│   │   ├── RGB
│   │   │   ├── *.[jpg|png|jpeg]
│   │   ├── depths
│   │   │   ├── *.[jpg|png|jpeg]
│   │   ├── GT
│   │   │   ├── *.[jpg|png|jpeg]
│   ├── test
│   │   ├── <dataset_name>
│   │   │   ├── RGB
│   │   │   │   ├── *.[jpg|png|jpeg]
│   │   │   ├── depths
│   │   │   │   ├── *.[jpg|png|jpeg]
│   │   │   ├── GT
│   │   │   │   ├── *.[jpg|png|jpeg]

```

## Note

- We can't overwrite existing split datasets -> Simply remove and push a new dataset

## Push to HF dataset

- Run `python main.py`

## Test dataset

- Run `python test.py`

## Download from Kaggle datasets

- Create a new token and put it in the file located at `~/.kaggle/kaggle.json`

- Download using Kaggle-CLI

```
kaggle datasets download <username/dataset_name>
```
