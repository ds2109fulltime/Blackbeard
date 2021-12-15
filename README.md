# Blackbeard Library. A python’s life for me!
This library is designed for people who need to optimize time in an agile way with an ease of understanding, as well as dealing with projects under the umbrella of Data Science, including cleaning data frame (including images), visualization and machine learning.

## Overview
This library consists of 3 parts that are the fundamental aspects of Data Science:

### Machine Learning
These functions are for ML-based projects and speed up their execution:
grid_search, models, NN_conv_model, model_train, load_model, save_model, basic_regression_models, regression_errors,print_regress_metrics, print_classif_metrics, split_and_scale, categorical_encoding, automl, captain.

### Visualization
These functions allow us to optimize the execution of the graphs in time:
draw_boxplot, draw_donut_chart, draw_maps, draw_three_countplot, show_roc_curve, sort, pairplot_heatmap, draw_missing_ratio, draw_statistic_values, draw_target_transformation, draw_feat_importance, draw_maps, draw_donut_chart, draw_three_countplot, show_roc_curve.

### Data Cleaning
Using the following functions, we are able to increase the efficiency when it comes down to cleaning data and images:
get_root_path, delete_duplicates, missing_values, save_col_names, Unix_to_UTC, download_csv, split_datetime, max_min, null_to_0, null_to_mean, categorical_numeric, df_information, merge_df, drop_NaN, eleate_column, replace_by_numpy, replace_string, percentage, sort_column, outliers, sales_by_category, month_to_name, columns_dtype,  get_products_that_are_sold_together, frames_from_video, images_dataset_properties, read_images, flip_images, resize_images
edit_images, color2gray, negative_colors_images, monocolor_images, images_load_data, save_images.

## Usage
In the following paragraphs, we are going to describe how you can use Blackbeard:

##  Getting it
To download Blackbeard, either fork this github repo or simply use Pypi via pip.
```sh
$ pip install BLACKBEARD_ds2109
```
## Using it
Blackbeard was programmed with ease-of-use in mind. First, import Blackbeard and then call any of the functions based on your needs.

If you are a Mac user, consider the following: When you are executing a function for images, by default, it installs the file called .DS_Store on your computer. In order for execute the function correctly, what you have to do is the following: right on click on the images folder that you downloaded, click on a terminal, and execute the following code:  find . -name '.DS_Store' -type f -delete.
