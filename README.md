# Operation Final Front++: Graph Extraction & Minimum Fuel Solver

Welcome! This repository contains the full pipeline to extract directed graphs from challenging images and solve the minimum fuel problem for Operation Final Front++.

```

    ___                     __       ____        __  __       
   /   |  _________  ____ _/ /_     / __ \____ _/ /_/ /_____ _
  / /| | / ___/ __ \/ __ `/ __ \   / / / / __ `/ __/ __/ __ `/
 / ___ |/ /  / / / / /_/ / /_/ /  / /_/ / /_/ / /_/ /_/ /_/ / 
/_/  |_/_/  /_/ /_/\__,_/_.___/  /_____/\__,_/\__/\__/\__,_/  

```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/DattaArnab/OperationFinalFront.git
cd OperationFinalFront
```

### 2. Set Up the Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:
- numpy
- opencv-python
- torch
- torchvision
- scikit-learn
- matplotlib
- pandas
- easyocr
- pytesseract


### 3. Run the pipeline using the pretrained model for digit classification

```bash
python fuel.py /path/to/image
```
For example
`python fuel.py graphs_images/1.png`

---

## Model Training (Redundant)

To train the digit classifier again:

```bash
python digit_classifier.py
```

This will train the CNN digit classifier and save the best model as `best_digit_classifier.pth`.

---




## Graph Extraction & Evaluation
To visualize the adjacency matrix of a particular image use the notebook `graph_visualize`

```bash
code graph_visualize.ipynb
```
- Visualizes the extracted graph from image and generates `graph_result.png`

To extract adjacency matrices from all images and evaluate accuracy use the notebook `save_adjacency_matrices` 

```bash
code save_adjacency_matrices.ipynb
```
- Processes all images in `graphs_images/`
- Outputs predicted adjacency matrices to `output.csv`
- Prints and saves extraction accuracy metrics to `accuracy_metrics.txt`

---

## Notes

- All code is tested on Ubuntu 22.04 with Python 3.10 with NVIDIA RTX 4050 Laptop GPU.
- No LLMs, VLMs, or other language models are used in the pipeline.
- For best results, use PNG images named as `<graph_id>.png`.

---

## Troubleshooting

- If you see missing dependencies, ensure you have run `pip install -r requirements.txt` inside the conda environment.
- For GPU acceleration, ensure you have a compatible CUDA setup and install the right PyTorch version.

---
