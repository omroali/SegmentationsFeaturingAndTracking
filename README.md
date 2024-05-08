
Please install all requirements into a python virtual environments
```
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

Task 1 Segmentation work can be found in seg_main.py, it relies on the `segmentation/image_segmentation.py` and `segmentation/utils.py` files to run. 
```
python seg_main.py
```
Segmentation operation happens in `segmentation/image_segmentation.py` in the `preprocessing` section. to display or hide a step, set the Bool variable for the `show` parameter in the step.
The relevant outputs can be found in the process_data folder

Task 2 Features work can be found in features_main.py
```
python features_main.py
```
The relevant outputs can be found in the `Report/assets/features` folder.

Task 3 Tracking work can be found in tracking_main.py
```
python tracking_main.py
```
The relevant output can be found in the `Report/assets/tracking` folder.
