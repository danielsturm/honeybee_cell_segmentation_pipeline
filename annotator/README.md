# Annotation Tool

- Uses Napari as framework (link)
- Has three main components the cell finder and the annotation tool and the segmentation mask creator

## Install dependencies
- I use poetry for dependency management, therefore this differs a bit if you are using a conda environment

### With Conda Envionment
- inside the conda environment:
    - `conda install -c conda-forge poetry`
    - `poetry config virtualenvs.create false` (since you are using a conda environment, poetry should not create it's own)
    - cd into annotator/
    - run `poetry install`

### With Poetry Environment
- install poetry: https://python-poetry.org/docs/#installation
- cd into annotator/
    - run `poetry install`

## How to use

### Cell finder
- Run the Cell Finder to find as many cells as possible beforehand.
- This will created a json file with the found cells, with the same name as the image

### Annotation Tool
- Each image needs a json file with pre detected cells
- If an image does not have a corresponding labels json, an empty one will be created
- start the annotation tool from annotator/ with `poetry run annotator`

- if you like you can enable tooltips for the cells layer that display the label and diameter
    - to to File -> Preferences -> Appearance -> Show layer tooltips

- select points by using the brush in the brush layer or the mouse in the points layer
- assign a new label to a point selection by using the dropdown
- use shortcuts to activate the points/cell layer (hit 'c') or the brush layer (hit 'b')
- use shortcuts to change to full/no opacity for the points in the points layer (hit 'y' for 0.0 opacity, 'x' for 1.0 opacity)
- change the point size by hitting Alt and scroll up or down (or use the slider from the tool bar)
- switch to next image with the arrows at the bottom

- if you need to change labels and/or respective colors, change label_map.json

### Segmentation Mask Creator
