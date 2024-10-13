# Content-Based Image Retrieval

This project implements a **Content Based Image Retrieval** system that searches for similar images based on color descriptors. The system is developed in Python as part of the course "Introduction to Human and Computer Vision."

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup Instructions](#setup-instructions)
    - [Clone the Repository](#1-clone-the-repository)
    - [Create Virtual Environment](#2-create-a-virtual-environment-with-venv)
    - [Install Dependencies](#3-install-dependencies)
    - [Organize the Datasets](#4-organize-the-datasets)
3. [Weekly development](#weekly-development)
    - [Week 1: Image Retrieval Based on Color Histograms](#week-1-image-retrieval-based-on-color-histograms)
    - [Week 2: Background Removal and Advanced Descriptors](#week-2-background-removal-and-advanced-descriptors)

---

## Project Structure

The project is organized into several folders with a modular structure to make it easy to extend and maintain:

```
Team3/
│
├── w1                          # Week 1 project folder
│   ├── src/                    # Source code for Week 1
│   ├── evaluation/             # Evaluation scripts
│   ├── data/                   # Folder for datasets (not included in the repo)
│   │   ├── BBDD/               # Folder with museum images
│   │   └── qsd1_w1/            # Folder with query images (QSD1)
│   ├── test_submission.py      # Submission test scripts for Week 1
│   └── utils/                 
│       ├── plot_results.py             
│       └── print_dict.py 
│
├── wX                           # Week X project folder (same structure as w1)
│   └── ...
│
...
|
├── README.md               
└── requirements.txt        
```

---

## Setup Instructions

### 1. Clone the Repository
First, clone this repository to your local machine:

```bash
git clone https://github.com/MCV-2024-C1-Project/Team3.git
cd Team3
```

### 2. Create a Virtual Environment with `venv`
To manage dependencies, create a virtual environment using Python’s built-in tool `venv`.

#### Create the virtual environment:
```bash
python -m venv env
```

#### Activate the virtual environment:
- **On Windows**: 
```bash
env\Scripts\activate
```
- **On macOS/Linux**: 
```bash
source env/bin/activate
```

### 3. Install Dependencies
With the virtual environment activated, install the project dependencies:

```bash
pip install -r requirements.txt
```

### 4. Organize the Datasets
Download the datasets from the course platform and place them in the `data/` folder as follows:
```
Team3/
├── wX/                         # Week X folder
│   └── data/                     
│       ├── qsd1_wX/            # Query images 1 for Week X
│       └── qsd2_wX/            # Query images 2 for Week X
```

---

## Weekly Development

### Week 1: Image Retrieval Based on Color Histograms
**Goal**: Retrieve the K most similar museum images for each query image based on color histograms.

#### Tasks:
1. **Task 1**: Compute image descriptors for both the museum dataset (BBDD) and the query dataset (QSD1). You can use up to two methods.  
   Example: Color Histograms (gray level or concatenated color component histograms in RGB, CieLab, YCbCr, HSV, etc.). It is compulsory to use 1D histograms.
2. **Task 2**: Implement different similarity measures to compare images, such as:
    - Euclidean distance
    - L1 distance
    - χ² distance
    - Histogram intersection
    - Hellinger kernel
3. **Task 3**: For each image in QSD1, compute the similarity to the museum images (BBDD) and return the top K images (highest score, lowest distance). Evaluation is performed using *mAP@K* (mean Average Precision at K).
4. **Task 4**: Submit your results for a "blind" competition in QST1. Create a Python list of lists with the image IDs (integers).  
   Example format:  
   ``` 
   Query: [[q1], [q2], [q3]]  
   Result: [[7,2], [76,4], [43,12]]  
   ```

#### To run the Week 1 system:
```bash
python w1/src/main.py
```
This will execute the image retrieval system for Week 1 using color histograms.


---

### Week 2: Background Removal and Advanced Descriptors
**Goal**: Retrieve paintings while handling images with background and implement more advanced histogram descriptors.

#### Tasks:
1. **Task 1**: Implement 3D / 2D and block and hierarchical histograms. Block-based histograms divide the image into non-overlapping blocks, compute histograms per block, and concatenate histograms. Spatial pyramid representation: compute block histograms at different levels and concatenate representations.
2. **Task 2**: Test the query system using query set QSD1-W2 (development) and evaluate retrieval results using your best performing descriptor. Compare results against the best descriptor from Week 1.
3. **Task 3**: For each image in the QSD2-W2, remove the background using the background color (e.g., model background distribution, use color thresholds). Create a binary mask to evaluate the method. Compute descriptors on the foreground pixels. Do not use contour detectors, object detectors, etc., just color!
4. **Task 4**: Evaluate the background removal using precision, recall, and F1-measure.
5. **Task 5**: For QSD2-W2, remove the background, apply the retrieval system, and return correspondences for each painting. Only retrieval is evaluated.

#### To run the Week 2 system:
```bash
python w2/src/main.py <query_images_folder_path> <dimension> <colorspace> <structure> <measure>
```
Where:
- `<query_images_folder_path>`: Path to the folder containing query images.
- `<dimension>`: 3D or 2D histogram.
- `<colorspace>`: HLS, HSV, etc.
- `<structure>`: block or hierarchical.
- `<measure>`: intersection, canberra, etc.

This will run the Week 2 retrieval system with the specified configurations.
