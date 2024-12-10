# SST Data Generation

This script requires the pandas module as a prerequisite.

```
python3 data/outputs/add_coralreefwatch_location_to_csv.py
```

This will add a field SST@90th_HS with the 90th percentile SST data based on the date to all metadata.csv files in the images directory.

# Regression model

The notebook for the regression model is at data/outputs/Regression.ipynb.
It is recommended to run the notebook in Google Colab with a GPU, but it will also work on any laptop with a powerful GPU.

# Results

The results from the regression model are at data/SST_Regression_Results.ipynb.

# Pytorch Class

Initial script is pytorch_class.ipynb, creating custom Class. It also includes a baseline, CNN, training in it.
denseNet.ipynb is modified on it for training.
Then dataset.py, main.py, models.py are created and modified.

# denseNet model

notebook script is at src/denseNet.ipynb
denseNet121 needed to be imported:
    from torchvision.models import DenseNet121_Weights
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
