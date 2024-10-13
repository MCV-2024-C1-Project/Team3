
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
4. [Deactivating the Virtual Environment](#deactivating-the-virtual-environment)

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

## Weekly development

### Week 1: Image Retrieval Based on Color Histograms
**Goal**: Retrieve the K most similar museum images for each query image based on color histograms.

#### Tasks:
1. **Image Descriptors**: Compute 1D color histograms for the museum (BBDD) and query datasets (QSD1).
2. **Similarity Measures**: Implement and compare similarity measures like Euclidean distance, histogram intersection, and others.
3. **Retrieval System**: Implement the retrieval system to return the top K results.
4. **Evaluation**: Submit the results for the QST1 blind test set.

#### To run the Week 1 system:
```bash
python w1/src/main.py
```
This will execute the image retrieval system for Week 1 using color histograms.

---

### Week 2: Background Removal and Advanced Descriptors
**Goal**: Retrieve paintings while handling images with background and implement more advanced histogram descriptors.

#### Tasks:
1. **3D/2D Histograms**: Implement both block-based and hierarchical histograms for retrieval.
2. **Background Removal**: Implement background removal techniques and compute descriptors only on foreground pixels.
3. **Evaluation**: Submit retrieval results for the QST2 dataset.

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

---

## Deactivating the Virtual Environment
After completing your work, deactivate the virtual environment:

```bash
deactivate
```

---

This structure should provide clear guidance on how to set up and run the project for both weeks, with organized tasks and descriptions for each week’s assignments.
