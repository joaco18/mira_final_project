# MEDICAL IMAGE REGISTRATION AND APPLICATIONS (MIRA)

## Final Project

### Kaouther Mouheb, Marwan Kefah, Joaquín Seia

This repository contains the code developed for the final project in MIRA course as part of the Join Masters Degree in Medical Imaging and Aplications (MAIA).

The repository structure is as follows:
    data -> contains the images to utilize
    dataset -> contains the dataset class developed to load the cases
    preprocessing -> preprocessing utils
    elastix -> contains the code for the elastix strategy
        parameter_maps -> elastix parameters files
        elastix_utils -> functions such as elastix wrappers
    dl -> deep learning strategy
        sth ->
    utils -> shared utils / metrics / plots
    notebooks -> exploratory or development notebooks

## Set up

To set up the repository run the following Unix promt comands:

### Environment
Create the environment

```bash
conda create -n mira_fp anaconda -y &&
conda activate mira_fp
```

Install requirements:
```bash
pip install -r requirements.txt
```

Add current repository path to PYTHONPATH
```bash
cwd=$(pwd) &&
export PYTHONPATH="${PYTHONPATH}:${cwd}/mira_final_project/"
```

### Data
- Option 1: Get the raw images and process them:

    Raw images can be downloaded from [this link](https://drive.google.com/file/d/1gc63UJqSrwcaQQKwD8R9KsIxtysZYzdf/view?usp=share_link). Or from [DIR-LAB](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html)

    The directory structure should be: mira_final_project/data/dir_lab_copd_raw/caseN/copdN_iBHCT.img
    
    Metadata json can be downloaded from [this link](https://drive.google.com/file/d/11QxECkvpHMwQcG90_r7Qu4fYm8wgNjea/view?usp=share_link).
    
    The directory structure should be: mira_final_project/data/dir_lab_copd_metadata.json
    
    To process the images run:
    ```bash
    python dataset/parse_raw_imgs.py
    ```

- Option 2: Get the processed images:

    Processed images can be download from [this link](https://drive.google.com/file/d/1OScdnhRwFZgIG7V47Jle2NYCseP5Uqmn/view?usp=share_link)

    The directory structure should be: mira_final_project/data/dir_lab_copd/caseN/copdN_iBHCT.img


# FOR WINDOWS
Intall elastix and be sure you can call it from command line.

# FOR LINUX
Just for linux
``` bash
cd elastix/ &&
cwd=$(pwd) &&
cd ../ &&
base="${cwd}/elastix-5.0.0-Linux" &&
# anacondaenv="/home/jseia/anaconda3/envs/mira_fp" &&
export PATH="${base}/bin":$PATH &&
export LD_LIBRARY_PATH="${base}/lib":$LD_LIBRARY_PATH 
```

In case elastix doesn't work, try removing the binary from main anaconda bin folder:
```
rm /home/jseia/anaconda3/bin/transformix
```

## Recommendations to contributers

The code in Medvision is developed following:
- numpy docstring format
- flake8 lintern
- characters per line: 100