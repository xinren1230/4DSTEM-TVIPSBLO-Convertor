# 4DSTEM TVIPS to BLO Convertor Wizard

This repository contains a PyQt5-based wizard application for processing TVIPS data to BLO file. The wizard guides the user through several steps—from parameter selection and mask checking, to virtual bright field (VBF) image generation, and finally conversion & batch processing of data files.

## Features

- **Parameter Selection & Mask Check**  
  Configure your dataset parameters and visually verify the mask position using a draggable mask.

- **Virtual BF Generation & Analysis**  
  Generate virtual bright-field images and analyze them to automatically suggest a starting frame.

- **Conversion & Batch Conversion**  
  Convert individual HDF5 files or process multiple files in a batch.  
  The interface allows you to add conversion tasks and manage additional files.

- **User-Friendly GUI**  
  A multi-step wizard built with PyQt5 that simplifies the 4DSTEM processing workflow.



## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.7 (or later)
- [PyQt5](https://pypi.org/project/PyQt5/)
- [NumPy](https://pypi.org/project/numpy/)
- [h5py](https://pypi.org/project/h5py/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [SciPy](https://pypi.org/project/scipy/)
- [scikit-image](https://pypi.org/project/scikit-image/)
- [Matplotlib](https://pypi.org/project/matplotlib/)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/4DSTEM-Processing-Wizard.git
   cd 4DSTEM-Processing-Wizard
   
conda create -n TVIPSBLO python=3.7
conda activate TVIPSBLO
pip install PyQt5 numpy h5py opencv-python scipy scikit-image matplotlib
python tvipsGUI.py

### Usage
Step 1: Parameter Selection & Mask Check

Select your TVIPS file and set required parameters.
Verify and adjust the mask using the draggable mask interface.
The suggested starting frame will be updated based on your input.


Step 2: Virtual BF Generation & Analysis

Generate the virtual BF image.
Analyze the image by clicking on it to update the starting frame.


Step 3: Conversion & Batch Conversion

Configure conversion parameters (e.g., image height, median, Gaussian, binning).


Use the Run Single Conversion button to convert an individual file.


Use the Add Task and Add File buttons to build a list of additional HDF5 files.


Finally, click Run Batch Conversion to process all the files in the list.
Progress and log messages are displayed in the interface.


### Contributing
Contributions, suggestions, and bug reports are welcome!


Please open an issue or submit a pull request to contribute.

### License
This project is licensed under the MIT License. See the LICENSE file for details.
