# SOCIB Beach wracks identification

[![Build Status](https://jenkins.cloud.ai4eosc.eu/buildStatus/icon?job=AI4OS-hub/socib-beach-wracks-identification/main)](https://jenkins.cloud.ai4eosc.eu/job/AI4OS-hub/job/socib-beach-wracks-identification/job/main/)

This module provides an AI-powered tool for instance segmentation of seagrass wracks from beach imagery. It automatically detects and segments beach wracks from RGB images and can distinguish between different densities of wrack accumulation. The module is integrated with [DEEPaaS API](https://github.com/ai4os/DEEPaaS) (platform support) enhancing the functionality and accessibility of the code, making it easier for users to leverage and interact with the pipeline efficiently.

The default model (yolo11m) was trained on [SOCIB](https://www.socib.es/)-derived data ([BWILD dataset](https://doi.org/10.5281/zenodo.12698763)), with its performance enhanced through data augmentation and hyperparameter optimization. The framework is extensible, allowing to use different yolo variants and sizes, fine-tune the models on custom datasets by adjusting hyperparameters, and using transfer learning.


# 🛠️ Install the API
To launch the API, first, install the package, and then run DeepaaS:
``` bash
git clone --depth 1 https://github.com/ai4os-hub/socib-beach-wracks-identification.git
cd  socib-beach-wracks-identification
pip install -e .
deepaas-run --listen-ip 0.0.0.0 --config-file deepaas.conf
```

><span style="color:Blue">**Note:**</span> Before installing the API, please make sure to install the following system packages: `gcc`, `libgl1`, and `libglib2.0-0` as well. These packages are essential for a smooth installation process and proper functioning of the framework.
```
apt update
apt install -y gcc
apt install -y libgl1
apt install -y libglib2.0-0
```

## 📂Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── socib_beach_wracks_identification
│   ├── README.md           <- Instructions on how to integrate your model with DEEPaaS.
│   ├── __init__.py         <- Makes socib-beach-wracks-identification a Python module
│   ├── config.py           <- Module to define CONSTANTS used across the AI-model python package
│   └── utils.py            <- Package to perform checks, pipelines, and other utilities
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data subpackage for the integration with DEEP API
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                 <- Folder to store your models. Includes the default SOCIB model weights
│   └── yolo11m_170325
│       └── weights
│           └──best.pt
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to install testing tools
├── requirements.txt        <- Requirements file to run the API and models
│
├── pyproject.toml         <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── models              <- Folder to store the models for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│   └── test_training       <- Tests folder for model training requirements
│
└── tox.ini                <- tox file with settings for running tox; see tox.testrun.org
```

# ⚙️ Environment variables settings
"In `./api/config.py` you can configure several environment variables:

- `DATA_PATH`: Path definition for the data folder; the default is './data'.
- `MODELS_PATH`: Path definition for saving trained models; the default is './models'.
- `REMOTE_PATH`: Path to the remote directory containing your trained models. Rclone uses this path for downloading or listing the trained models.
- `YOLO_DEFAULT_WEIGHTS`: Define default timestamped weights for your trained models to be used during prediction. If no timestamp is specified by the user during prediction, or it is set to None, the first model in YOLO_DEFAULT_WEIGHTS will be used. Format them as "timestamp1, timestamp2, timestamp3, ..."

# 📊 Track Your Experiments with MLflow
If you want to use Mflow to track and log your experiments, you should first set the following environment variables:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `MLFLOW_EXPERIMENT_NAME` (for the first experiment)

optional options:

- `MLFLOW_RUN`
- `MLFLOW_RUN_DESCRIPTION`
- `MLFLOW_AUTHOR`
- `MLFLOW_MODEL_NAME`: This name will be used as the name for your model registered in the MLflow Registry.
- Then you should set the argument `Enable_MLFLOW` to `True` during the execution of the training.


# 📁 Dataset Preparation
- To train the socib-beach-wracks-identification module on your own dataset, your images may have different formats (e.g., PNG, JPG) but your annotations must be saved in yolo format (.txt). Please organize your data in the following structure:
```
socib_beach_wracks_identification
│
└── data
    └── processed
        ├──  images
        │    ├── train
        │    │   ├── img1.jpg
        │    │   ├── img2.jpg
        │    │   ├── ...
        │    ├── val
        │    │   ├── img10.jpg
        │    │   ├── img11.jpg
        │    │   ├── ...
        |    ├── test
        │    │   ├── img20.jpg
        │    │   ├── img21.jpg
        │    │   ├── ...
        │    
        ├── labels    
        │    ├── train
        │    │   ├── img1.txt
        │    │   ├── img2.txt
        │    │   ├── ...
        │    ├── val
        │    │   ├── img10.txt
        │    │   ├── img11.txt
        │    │   ├── ...
        |    ├── test
        │    │   ├── img20.txt
        │    │   ├── img21.txt
        │    │   ├── ...
        │    
        └── data.yaml
```

The `data.yaml` file contains the following information about the data:

```yaml
path: 'path_to_my_dataset_folder' #dataset root dir
train: images/train #training split (relative to 'path')
val: images/val #validation split (relative to 'path') 
test: images/test #testing split (relative to 'path') - Optional

# Classes
names:
  0: dense wrack
  1: intermediate wrack
    

```
The `train`, `val` and `test` fields specify the paths to the directories containing the training, validation, and testing images, respectively.
`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

><span style="color:Blue">**Note:**</span> The default model was trained with rather small images (640x480) to avoid downsampling in YOLO training. A sample of the data and the folder structure for training the model is provided in `/socib_beach_wracks_identification/tests/data/seg`. To re-train the model with your own data, place it under `/socib_beach_wracks_identification/data/processed`. 

# 🚀 Launching the API

To train the model, run:
```
deepaas-run --listen-ip 0.0.0.0
```
Then, open the Swagger interface, change the hyperparameters in the train section, and click on train.

><span style="color:Blue">**Note:**</span>  Please note that the model training process may take some time depending on the size of your dataset and the complexity of your custom backbone. Once the model is trained, you can use the API to perform inference on new images.

><span style="color:Blue">**Note:**</span> Augmentation Settings:
Among the training arguments, there are options related to augmentation, such as flipping, scaling, etc. The default values are set to automatically activate some of these options during training (recommended). If you want to tune the augmentation settings, please review the default values and adjust them accordingly.

# 🔍 Inference Methods

You can utilize the Swagger interface to upload your images or videos and obtain the following outputs:

- For images:

    - An annotated image highlighting the object of interest with a bounding box.
    - A JSON string providing the coordinates of the bounding box, the object's name within the box, and the confidence score of the object detection.

- For videos:

    - A video with bounding boxes delineating objects of interest throughout.
    - A JSON string accompanying each frame, supplying bounding box coordinates, object names within the boxes, and confidence scores for the detected objects.

## 📚 References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [MLflow – Open Source Experiment Tracking](https://mlflow.org/)
- [BWILD – Beach seagrass Wrack Identification Labelled Dataset](https://doi.org/10.5281/zenodo.12698763)
- [Soriano-González, J., et al. (2025) – Machine learning-driven shoreline extraction and beach seagrass wrack detection from beach imaging systems](https://coastaldynamics25.web.ua.pt/) 
