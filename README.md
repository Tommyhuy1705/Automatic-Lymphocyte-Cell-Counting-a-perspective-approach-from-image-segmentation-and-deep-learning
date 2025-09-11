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

## **Usage**
**- To train, evaluate, and analyze the YOLO and U-Net models for lymphocyte cell counting, follow the steps below:**

---

## **Results**
### Key Findings:


___
 
---

## Model Training Notebooks
### Datasets Usage:
- Cell Counting (Roboflow): [Use for train YOLO & U-net model](https://www.kaggle.com/datasets/tensura3607/cell-counting-roboflow-segmentation-masks)

### U-Net
[View tSSN Training Notebook on Kaggle]()
>This notebook contains full training pipeline for U-Net segmentation model.

### YOLOv8
[YOLOv8 Training Notebook on Kaggle]()
>This notebook covers YOLO-based training for cell detection and counting.

## Notes
- Models were trained on Kaggle GPU environments.

---

## **Contributions**

- Designed and implemented a pipeline for automated lymphocyte cell counting.
- Integrated YOLO for detection-based counting.
- Developed and trained U-Net for segmentation-based counting.
- Applied a data augmentation pipeline (rotation, flipping, shifting, scaling, brightness, noise).
- Evaluated models with MSE, MAE, MAPE, and R² for robust benchmarking.
- Highlighted strengths and limitations of detection vs segmentation approaches.
- Structured the project with modular code, reproducible notebooks, and clear dataset usage.

---

## **Future Work**
- Expand dataset size and diversity for better model generalization.
- Explore hybrid detection + segmentation models for improved accuracy.
- Test lightweight architectures (e.g., MobileNet-based U-Net) for faster inference.
- Incorporate uncertainty estimation for more reliable cell counting.
- Benchmark against additional metrics such as F1-score for detection.
- Investigate transfer learning from large-scale biomedical segmentation datasets.

---

## **Acknowledgments**
Special thanks to the contributors and open-source community for providing tools and resources.

--- 


