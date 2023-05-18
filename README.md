# GazeScorer README


## Video Processing - Gazescorer

Gazescorer is a python library that can be used for the analysis of videos to estimate the gaze orientation of participants. 

##### Installation

To install the required packages to run Gazescorer, the simplest way is to create a Conda environment using the provided enviroment file via:

```sh
conda env create -f BRM_GS.yml
```

Once the environment has installed, activate it by running:
```sh
conda activate BRM_GS
```

If you would prefer to install the packages individually the required packages are below. These can generally be installed using either Conda or pip with the exception of FFmpeg and ffmpeg-python which are best to be installed using anaconda to ensure the correct version is installed. 

##### Dependencies:
- python=3.10
- cmake
    - Must be installed before dlib 
- dlib
- ipykernel
- ffmpeg v4.4.2 
    - For the correct version I recomend using:
        ```sh
        conda install -c anaconda ffmpeg
        ```
- ffmpeg-python
    - Install after ffmpeg to avoid conflicts. I would recomend installing using:
        ```sh
        conda install -c conda-forge ffmpeg-python
        ```
 - pandas
 - matplotlib
 - scipy
 - imutils
 - plotnine
 - sklearn
 - opencv-python

 It is also important that the **shape_predictor_68_face_landmarks.dat** file is in the parent directory of the GazeScorer directory. 

#### Running the analysis pipeline


To demonstrate the analysis pipeline three example videos have been provided. The analysis pipeline is in the jupyter notebook **Example_BRM.ipynb**. The first cell will process the three raw videos contained in the *input/example_video/* directory and append the scoring in the *input/example_scoring* directory. It will save the processed videos in the directory *input/example_vidoe/proccesed/*, the output data in the folder *output/example_datasets/*, and a plot of the detected face landmarks in the directory *output/face_lmarks/*.

The subsequent cells in the notebook will plot some simple inter-rater reliability outputs for both the static and dynamic phases. 



## Documentation

A basic documentation has been started and can be found in the docs folder