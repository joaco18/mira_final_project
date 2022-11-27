# MEDICAL IMAGE REGISTRATION AND APPLICATIONS (MIRA)

## Final Project

### Kaouther Mouheb, Marwan Kefah, Joaquín Seia

This repository contains the code developed for the final project in MIRA course as part of the Join Masters Degree in Medical Imaging and Aplications (MAIA).

The repository structure is as follows:
    data -> contains the images to utilize
    dataset -> contains the dataset class developed to load the cases
    preprocessing -> preprocessing utils
    elastix -> contains the code for the elastix strategy
        runnable -> executable file to run elastix
        registration_utils -> useful functions to register images such as elastix wrappers
    dl -> deep learning strategy
        sth ->
    utils -> shared utils
    notebooks -> exploratory or development notebooks

## Set up

To set up the repository run the following Unix promt comands:

### Environment
Create the environment

```bash
conda create -n mira_fp python==3.9.13 anaconda -y &&
conda activate mira_fp &&
conda update -n mira_fp conda -y
```

Install requirements:
```bash
pip install -r requirements.txt
```

Add current repository path to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:[PATH/mira_final_project/]"
```

### Data
- Option 1: Get the raw images and process them:

    Raw images can be download from [this link](https://drive.google.com/file/d/1bI3pi2iPcYYPlWtnW2ibHaUxZQjESauf/view?usp=share_link). Or from [DIR-LAB](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html)

    The directory structure should be: mira_final_project/data/dir_lab_copd_raw/caseN/copdN_iBHCT.img

    To process the images run:
    ```bash
    python parse_raw_imgs.py
    ```

- Option 2: Get the processed images:

    Processed images can be download from [this link](https://drive.google.com/file/d/1bI3pi2iPcYYPlWtnW2ibHaUxZQjESauf/view?usp=share_link)

    The directory structure should be: mira_final_project/data/dir_lab_copd/caseN/copdN_iBHCT.img


## Recommendations to contributers

The code in Medvision is developed following:
- numpy docstring format
- flake8 lintern
- characters per line: 100