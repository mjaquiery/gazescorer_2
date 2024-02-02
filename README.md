# Current workflow (L2CS)

This workflow uses the [L2CS](https://github.com/Ahmednull/L2CS-Net) model to predict gaze direction from images of faces.
Data (not provided) are videos of people's faces recorded while they perform a variety of tasks in which the response is to determine which side of the screen contains a target object.
The target may be, for instance, the larger of two numbers.

The workflow is as follows:

- copy data to `./video_input/[dataset]` where `[dataset]` is the name of the dataset
  - our examples are `BRM_input` and `Number_comparison_online_new`
  - there are also `left`, `right`, and `trial` directories for an eyetracking dataset
- run the relevant `L2CS` script, preferably on a GPU
  - you may need to tweak the `device` argument to the `Pipeline` class to suit your setup
  - run `L2CS_run_gorilla.py` for the `BRM_input` dataset
  - run `L2CS_run_NCO.py` for the `Number_comparison_online_new` dataset
  - run `L2CS_run_eyetracking.py` for the eyetracking dataset
- analyse results in R
  - use `renv::activate()` and `renv::restore()` to set up the R environment
  - run `analysis/L2CS_gorilla.Rmd` for the `BRM_input` dataset
  - run `analysis/L2CS_NCO.Rmd` for the `Number_comparison_online_new` dataset
  - run `analysis/L2CS_with_ET.Rmd` for the eyetracking dataset

There are various other `.py` files that are hangovers from exploring L2CS functionality.  They are not currently used in the workflow.
For new datasets, you will need to create a new `L2CS` script, and possibly a new R analysis script.

This project has evolved slowly over time, and has been used for a variety of datasets.  
The current workflow is the result of a long process of trial and error, and could be replaced by a more generalised workflow in the future.

# Archived workflow

The following is a workflow that was used to train models for a research project.  It is not currently in use, but is kept here for reference.

During this workflow, we used the Columbia Gaze Data set to train models to predict gaze direction from images of faces.  

- download Columbia Gaze Data set
- extract to `./columbia_gaze_data_set`
- run `extract_eyes.py`
- run `prepare_model_data.py`
- run `train_models.py`
- analyse results with `analysis/simple_accuracy_analysis.Rmd`
