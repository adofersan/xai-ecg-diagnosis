# Explainable AI for Automatic Heart Disease Diagnosis Using 3DFMMecg Features: A Novel ECG-Based Approach

## Project Structure

This project is structured to facilitate the training and evaluation of machine learning models. Below is a brief description of the main directories and files:

### Hidden Folders on Github private repository
- **data/**: Contains datasets and related files (not uploaded to the repository due to privacy reasons).
- **img/**: Image assets (not uploaded because they are placed in the report and they are too heavy).
- **venv/**: Virtual environment directory (you have to create it).
- **models/**: Directory for storing trained models (not uploaded because they are too heavy).

### Actual Folders and Files in ZIP file

- **src/**: Source code for model training and evaluation.

  - **feature_engineering.ipynb**: Jupyter notebook for extracting features.
  - **labels.py**: Python script containing class label definitions.
  - **metrics_multilabel.py**: Python script containing the multilabel metrics (instance-AUC metrics were not used at the end).
  - **model_training.ipynb**: Jupyter notebook for training models.
  - **PTBXLModel.py**: Python script containing a class for modelling the PTBXL tasks.
  - **tables_and_figures.ipynb**: Jupyter notebook for generating the figures and tables of the report.

- **requirements.txt**: List of dependencies required for the project.

## Setup

To set up the project, follow these steps:

1. Create a virtual environment:
   ```sh
   python -m venv venv
   ```

2. Activate the virtual environment:
    
    - On Windows:

    ```
    venv\Scripts\activate
    ```

    - On macOS/Linux:

    ```
    source venv/bin/activate
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

