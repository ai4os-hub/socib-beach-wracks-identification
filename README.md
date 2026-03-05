# SOCIB Beach wracks identification

[![Build Status](https://jenkins.cloud.ai4eosc.eu/buildStatus/icon?job=AI4OS-hub/socib-beach-wracks-identification/main)](https://jenkins.cloud.ai4eosc.eu/job/AI4OS-hub/job/socib-beach-wracks-identification/job/main/)

Developed by [SOCIB](https://www.socib.es/), this module offers an AI-driven solution for the instance segmentation of seagrass wracks within beach imagery. It automatically detects and segments beach wracks from RGB images and can distinguish between different densities of wrack accumulation. The module is integrated with [DEEPaaS API](https://github.com/ai4os/DEEPaaS) (platform support) enhancing the functionality and accessibility of the code, making it easier for users to leverage and interact with the pipeline efficiently.

![beach wracks_output_example](https://raw.githubusercontent.com/ai4os-hub/socib-beach-wracks-identification/main/reports/figures/bwracks_output_example.png)

The underlying model (yolo11m-seg) was trained on the [BWILD dataset](https://doi.org/10.5281/zenodo.12698763), with its performance enhanced through data augmentation and hyperparameter optimization. To maintain accuracy when processing large images, particularly for detecting small wracks, the framework incorporates [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi) that enables sliced inference (i.e. image-patch based prediction), effectively supporting variable image resolutions with minimal accuracy reduction.

## 🚀 Running the container

### ☁️ Directly from Docker Hub

To run the Docker container directly from Docker Hub and start using the API, simply run the following command:

```bash
docker run -ti -p 5000:5000 ai4oshub/socib-beach-wracks-identification
```

This command will pull the Docker container from the Docker Hub [ai4oshub](https://hub.docker.com/u/ai4oshub/) repository and start the default command (`deepaas-run --listen-ip=0.0.0.0`).

**N.B.** For either CPU-based or GPU-based images you can also use [udocker](https://github.com/indigo-dc/udocker).

### 🛠️ Building the container

To build the container directly on your machine (for example, if you need to modify the `Dockerfile`), use the instructions below:
```bash
git clone https://github.com/ai4os-hub/socib-beach-wracks-identification
cd socib-beach-wracks-identification
docker build -t ai4oshub/socib-beach-wracks-identification .
docker run -ti -p 5000:5000 ai4oshub/socib-beach-wracks-identification
```

These three steps will download the repository from GitHub and will build the Docker container locally on your machine. You can inspect and modify the `Dockerfile` in order to check what is going on. For instance, you can pass the `--debug=True` flag to the `deepaas-run` command, in order to enable the debug mode.

## 🔌 Connect to the API

Once the container is up and running, browse to http://0.0.0.0:5000/ui to get the [OpenAPI (Swagger)](https://www.openapis.org/) documentation page.

## 📂 Project structure
```
.
├── Dockerfile                        # Commands to build the Docker image
├── Jenkinsfile                       # CI/CD pipeline configuration
├── LICENSE
├── README.md
├── VERSION
├── ai4-metadata.yml                  # Metadata for the AI4OS platform
├── api                               # DEEPaaS API integration logic
│   ├── config.py
│   ├── responses.py
│   ├── schemas.py
│   └── utils.py
├── data                              # Directory for storing datasets
│   ├── processed
│   └── raw
├── deepaas.conf                      # Configuration file for DEEPaaS
├── docs                              # Sphinx documentation project
├── models                            # Pre-trained models directory
│   └── yolo11m_170325                # YOLOv11 specific model folder
│       └── weights
│           └── best.pt               # Best model weights
├── pyproject.toml                    # Project configuration and dependencies
├── reports                           # Generated reports and figures
│   └── figures
├── requirements.txt                  # Dependencies to run the API and models
├── socib_beach_wracks_identification # Main Python package source code
│   ├── config.py
│   └── utils.py
├── tests                             # Unit and integration tests
└── tox.ini                           # Test automation configuration
```

## 🇪🇺 Acknowledgements

This work was supported by ‘iMagine’ (Grant Agreement No.101058625) and ‘FOCCUS’ (Grant Agreement No.101133911) European Union funded projects. Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA).

## 📚 References

- [BWILD – Beach seagrass Wrack Identification Labelled Dataset](https://doi.org/10.5281/zenodo.12698763)
- [Soriano-González, J., et al. (2025) – Machine learning-driven shoreline extraction and beach seagrass wrack detection from beach imaging systems](https://coastaldynamics25.web.ua.pt/)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Akyon, F. et al. (2022) - Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://doi.org/10.1109/ICIP46576.2022.9897990)
