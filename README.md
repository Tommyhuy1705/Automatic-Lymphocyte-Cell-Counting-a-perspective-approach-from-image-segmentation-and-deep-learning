# Benchmarking-YOLO-and-U-Net-Segmentation-Models-for-Automated-Lymphocyte-Cell-Counting
## Introduction  

Accurate lymphocyte cell counting plays a critical role in biomedical research and clinical diagnostics, providing essential insights into immune system health and disease progression. Traditionally, this task has been performed manually by experts under microscopes, which is both time-consuming and prone to human error. To address these challenges, automated computer vision techniques have emerged as promising solutions for reliable and scalable cell analysis.

In this project, we benchmark two popular deep learning approaches for lymphocyte cell counting:

- YOLO (You Only Look Once) — a state-of-the-art object detection model widely used for fast and precise localization of individual objects in images.
- U-Net — a convolutional neural network architecture specifically designed for biomedical image segmentation, effective at capturing pixel-level structures.

We evaluate both models on an augmented dataset of lymphocyte cell images, focusing on accuracy, error metrics, and robustness. Our goal is to compare the strengths and weaknesses of detection-based (YOLO) versus segmentation-based (U-Net) approaches, providing practical insights into their suitability for automated lymphocyte analysis.

This study not only demonstrates the feasibility of deep learning in biomedical image processing but also highlights important considerations for selecting the right architecture depending on application requirements such as precision, speed, and scalability.

## **Features**
- Automated lymphocyte cell counting for biomedical image analysis.
- Cell localization and detection using YOLO.
- Pixel-level segmentation using U-Net.
- Comparison of two approaches:
  - Detection-based (bounding box predictions with YOLO)
  - Segmentation-based (binary masks with U-Net)
- Evaluation with regression and error metrics:
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**
  - **Mean Absolute Percentage Error (MAPE)**
  - **Coefficient of Determination (R²)**
- Data augmentation pipeline (rotation, flipping, shifting, scaling, brightness adjustment, Gaussian noise).
- Benchmarking on augmented lymphocyte dataset with consistent training/testing splits.
- Practical insights on model suitability for accuracy, speed, and scalability in biomedical workflows.

---

## Project Structure  
```plaintext

```

---

## **Installation**
Follow the steps below to set up the project:

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/Tommyhuy1705/Benchmarking-YOLO-and-U-Net-Segmentation-Models-for-Automated-Lymphocyte-Cell-Counting.git
   cd Benchmarking-YOLO-and-U-Net-Segmentation-Models-for-Automated-Lymphocyte-Cell-Counting
   ```

2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

---

## **Kaggle API Token Setup**

To access and download datasets directly from Kaggle within this project, follow these steps to set up your Kaggle API token:

1. Go to your [Kaggle account settings](https://www.kaggle.com/account).
2. Scroll down to the **API** section.
3. Click on **"Create New API Token"** – a file named `kaggle.json` will be downloaded.
4. Place the `kaggle.json` file in the root directory of this project **or** in your system's default path:  
   - Linux/macOS: `~/.kaggle/kaggle.json`  
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
5. Make sure the file has appropriate permissions:  
   ```bash
   chmod 600 ~/.kaggle/kaggle.json

---

## **Usage**
**- To train, evaluate, and analyze the Triplet Siamese Signature Network (tSSN), follow the steps below:**
1. **Prepare your input dataset:**
- Place raw document images into a folder `(e.g. data/raw_documents/)` if you're using YOLO for signature localization.
2. **Localize signature regions using YOLOv10:**
- Open and run the notebook `notebooks/yolov10-bcsd_training.ipynb`.
- This will detect and crop the signature regions from input documents and save them into a designated output directory `(e.g. data/signatures/)`.
3. **Configure model settings and experiment parameters:**
- Open `configs/config_tSSN.yaml`.
- Modify parameters as needed:
  - `distance_mode: choose from euclidean, cosine, manhattan, learnable`
  - `margin: set values like 0.2, 0.4, ..., 1.0`
  - `feature_dim, batch_size, epochs, and other hyperparameters`
4. **Train the Triplet Siamese Network (tSSN)**
- Open and run the notebook `notebooks/model_training.ipynb`.
- The training loop will:
  - Use `tSSN_trainloader.py` for balanced triplet sampling.
  - Build the model from `Triplet_Siamese_Similarity_Network.py`.
  - Apply the selected loss from `triplet_loss.py`.
5. **Evaluate model performance:**
- Run the notebook `experiment/Evaluation-cells-counting.ipynb` to:
- Compute accuracy, MSE, MAE.
- Visualize evaluation metrics.

---

## **Results**
### Key Findings:
1. **Best-performing configuration:**
- Triplet Network with Euclidean distance and margin = 0.6
- Accuracy: 95.6439% on CEDAR dataset
2. Learnable distance function showed potential but did not outperform fixed metrics.
3. Balanced batch sampling improved generalization across user styles.
4. Embedding visualizations show clear separation between genuine and forged signatures.

___
 
---

## Model Training Notebooks
### Datasets Usage:
- Cell Counting (Roboflow): [Use for train YOLO & U-net model](https://www.kaggle.com/datasets/tensura3607/cell-counting-roboflow-segmentation-masks)

### triplet Siamese Similarity Network (tSSN)
[View tSSN Training Notebook on Kaggle](https://www.kaggle.com/code/giahuytranviet/triplet-trainmodel)
>This notebook contains the full training process for the tSSN model, including preprocessing, training.

### YOLOv10
[YOLOv10 Training Notebook on Kaggle](https://www.kaggle.com/code/nguyenthien3001/yolov10-bcsd)
>This notebook covers training the YOLOv10 model for object detection, including data loading, training, and inference demo.

## Notes
- Models were trained on Kaggle GPU environments.

---

## **Contributions**

- Designed and implemented the full pipeline for offline signature verification using a Triplet Siamese Network (tSSN).
- Integrated YOLOv10 for efficient signature region localization from scanned documents.
- Developed flexible Triplet Loss module supporting multiple distance metrics: Euclidean, Cosine, Manhattan, and Learnable.
- Implemented a balanced batch sampler to improve triplet selection and training stability.
- Conducted extensive experiments with margin tuning and distance metric variations.
- Achieved 95.6439% accuracy on the CEDAR dataset using Euclidean distance with margin = 0.6.
- Visualized performance through ROC curves, precision-recall metrics, and embedding space analysis.
- Structured the project for reproducibility and scalability, using modular PyTorch components and well-documented notebooks.
- Prepared supporting materials including dataset configuration, training logs, and evaluation tools.

---

## **Future Work**
- Cross-dataset evaluation on GPDS, BHSig260 for generalizability.
- Integrate lighter backbones (e.g., MobileNet) for real-time performance.
- Incorporate attention mechanisms for enhanced local feature focus.
- Explore adaptive or learnable margin strategies.
- Apply to multilingual and multicultural signature styles.
- Introduce explainable AI components for visualizing decision-making process.

---

## **Acknowledgments**
Special thanks to the contributors and open-source community for providing tools and resources.

--- 


