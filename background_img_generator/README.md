# Background Image Generator
- The generator uses the old segmentation model to segment the bees from https://github.com/IvanMatoshchuk/honeybee_cells_segmentation_inference

## Install dependencies
- I use poetry for dependency management, therefore this differs a bit if you are using a conda environment

### With Conda Envionment
- inside the conda environment:
    - `conda install -c conda-forge poetry`
    - `poetry config virtualenvs.create false` (since you are using a conda environment, poetry should not create it's own)
    - cd into background_img_generator/
    - run `poetry install`

### With Poetry Environment
- install poetry
- cd into background_img_generator/
    - run `poetry install`

- the frame extractor requires a binary of ffmpeg with cuda support.
    - please download it and put the bin/exe into background_img_generator/frame_extractor/bin
    - (windows) https://www.gyan.dev/ffmpeg/builds/#git-master-builds latest git master branch build
    - (linux) https://github.com/BtbN/FFmpeg-Builds/releases (e.g ffmpeg-master-latest-linux64-gpl.tar.xz)

### Note: Possibly reinstall pytorch in environment
- unfortunately some packages like monai or albumentations tend to downgrade pytorch, effectively removing cuda support.
- I after installing all dependencies i had to install pytorch again.
- run (your cuda version might be different) `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Run Background Image Generator
- you need to set up 3 directories:
    1. input for the extractor: contains input directories per day
    2. intermediate directory: to store the extracted frames (this needs to be passed as output dir to the extractor and input dir to the bg image generator)
    3. output dir for the bg img generator: you will find the generated background images inside each camera dir
- inside background_img_generator: run `python main.py`
- you can run the whole module again and it will find the images where it stopped last time.
- see the documentation string of the BgImageGenConfig for more infos about the arguments
    - most of the arguments of the config probably don't need to be changed.
    - num_median_images is the most interesting one. can be set to e.g. 200
