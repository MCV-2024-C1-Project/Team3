# Content Based Image Retrieval

This project implements a **Content Based Image Retrieval** system that searches for similar images based on color descriptors. The system is developed in Python as part of the course "Introduction to Human and Computer Vision."

## Project Structure

The project is organized into several folders with a modular structure to make it easy to extend and maintain:

```
Team3/
│
├── data/                   # Folder for datasets (not included in the repo)
│   ├── BBDD/               # Folder with museum images
│   ├── qsd1_w1/            # Folder with query images (QSD1)
│   └── qsd2_w1/            # Folder with query images with background(QSD2)
│
├── w1                      # Week1 project folder
│   ├── evaluation/                 
│   │   ├── bbox_iou.py    
│   │   ├── average_precision.py        
│   │   └── evaluation_funcs.py     
│   │
│   ├── src/                     # Main source code
│   │   ├── descriptors.py       # Functions to compute image descriptors
│   │   ├── similarity.py        # Functions to compute similarity measures
│   │   └── main.py              # Main script to run the entire pipeline
│   ├── geometry_utils.py
│   ├── score_painting_retrieval.py
│   └── test_submission.py
│
├── w2                      # Week2 project folder
│   ├── evaluation/                 
│   │   ├── bbox_iou.py    
│   │   ├── average_precision.py        
│   │   └── evaluation_funcs.py     
│   │
│   ├── src/                     # Main source code
│   │   ├── background_removal.py       # Functions to remove the background
│   │   ├── descriptors.py              # Functions to compute image descriptors
│   │   ├── similarity.py               # Functions to compute similarity measures
│   │   └── main.py                     # Main script to run the entire pipeline
│   ├── geometry_utils.py
│   ├── score_painting_retrieval.py
│   └── test_submission.py
│
├── utils/                 
│   ├── plot_results.py             
│   └── print_dict.py   
│
├── README.md               
└── requirements.txt        


```

## Setup Instructions

To set up the environment and dependencies for the project, follow these steps:

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/MCV-2024-C1-Project/Team3.git
cd Team3
```

### 2. Create a Virtual Environment with `venv`

To manage dependencies, we will create a virtual environment using Python’s built-in tool `venv`.

#### Create the virtual environment:

Run the following command in the root of your project directory to create a new virtual environment called `env`:

```bash
python -m venv env
```

This will create a new folder `env/` in your project directory where the virtual environment will be stored.

#### Activate the virtual environment:

- **On Windows**:
  ```bash
  env\Scripts\activate
  ```

- **On macOS/Linux**:
  ```bash
  source env/bin/activate
  ```

When the virtual environment is activated, you’ll see `(env)` at the beginning of your terminal prompt.

### 3. Install Dependencies

With the virtual environment activated, install the project dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Organize the Datasets

Since the datasets are too large to be uploaded to GitHub, you will need to manually download them from the course platform.

1. **Download the datasets**:
   - **BBDD**: Museum image collection.
   - **qsd1_w1**: Query images for evaluation.

2. **Place the datasets in the `data/` folder**:
   - Museum images should go into `data/BBDD/`.
   - Query images should go into `data/qsd1_w1/`.

   Your `data/` folder should look like this:

   ```
   Team3/
   ├── data/
   │   ├── BBDD/
   │   │   ├── 00001.jpg
   │   │   ├── 00002.jpg
   │   │   └── ...
   │   └── qsd1_w1/
   │       ├── 00001.jpg
   │       ├── 00002.jpg
   │       └── ...
   ```

### 5. Run the Project

Once the datasets are organized and the dependencies are installed, you can run the project as follows.

To execute Week1 project, run this command

```bash
python w1/src/main.py
```

This will run the image retrieval system, generate the descriptors, compute the similarities, and output the results.

To execute Week2 project, run this command

```bash
python w2/src/main.py <query_images_folder_path>
```

This will run the image retrieval system, generate the descriptors, remove the background if neccesary, compute the similarities, and output the results.

### 7. Deactivate the Virtual Environment

When you are done working on the project, you can deactivate the virtual environment by running:

```bash
deactivate
```

This will return your terminal to its default Python environment.




