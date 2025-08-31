# 🚀 GlucoGuard: A Deep Learning Pipeline for Glucose Prediction 🩸

## 🌟 Summary

Welcome to GlucoGuard! This project is a comprehensive, state-of-the-art deep learning pipeline for continuous glucose monitoring (CGM) data analysis and prediction. Using a sophisticated transformer-based model, this pipeline processes CGM data, engineers a rich set of features, and predicts future glucose levels to provide proactive alerts for hypoglycemia and hyperglycemia.

This project is built with PyTorch and PyTorch Lightning, and it uses the `iglu-py` library for robust and clinically validated glycemic variability metrics.

## 📁 File Structure

Here is a breakdown of the project's file structure:

*   **`main.py`**: The main entry point for running the training pipeline. 🏃‍♂️
*   **`test_and_demo.py`**: A script to generate synthetic data and run a smoke test of the entire pipeline. Perfect for quick verification and experimentation! 🧪
*   **`requirements.txt`**: A list of all the Python packages needed to run the project. 📦
*   **`src/`**: The main source code directory.
    *   **`data_processing/`**: Modules for handling data loading and preprocessing.
        *   **`adapter.py`**: The `AwesomeCGMAdapter` class, which handles loading data from the Awesome-CGM datasets.
        *   **`cgm_dataset.py`**: The `AdvancedCGMDataset` class, which is the core of the data pipeline. It takes CGM data, engineers over 100 features, and prepares it for the model.
    *   **`features/`**: Modules for feature engineering.
        *   **(This directory is currently empty as feature calculations are handled in `cgm_dataset.py` using `iglu-py`)**
    *   **`models/`**: The deep learning model architecture.
        *   **`building_blocks.py`**: Contains the core components of the model, like `MultiHeadTemporalAttention` and `GatedResidualNetwork`.
        *   **`predictor.py`**: The main `HierarchicalGlucosePredictor` model, which assembles the building blocks into a powerful transformer-based architecture.
    *   **`training/`**: Modules for training the model.
        *   **`losses.py`**: The custom `GlucosePredictionLoss` function, which includes clinically-aware penalties for more accurate training.
        *   **`lightning_module.py`**: The `GlucoseLightningModule`, which wraps the model, data, and training logic together for easy training with PyTorch Lightning.
    *   **`inference/`**: Modules for running the model in a real-time setting.
        *   **`monitor.py`**: The `RealTimeGlucoseMonitor` and `AlertManager` classes, which simulate a real-time monitoring and alerting system.
        *   **`postprocessing.py`**: The `EnsemblePredictor` and `PostProcessor` classes, for improving and stabilizing predictions.

## 🚀 How to Use

### 1. 📦 Installation

First, you need to install all the required packages. It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (if you have an NVIDIA GPU)
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install the rest of the packages
pip install -r requirements.txt
```

### 2. 💾 Download the Data

This pipeline is designed to work with the [Awesome-CGM datasets](https://github.com/IrinaStatsLab/Awesome-CGM). You will need to download the datasets you want to use and place them in the `cgm_data` directory (or a directory of your choice). The `AwesomeCGMAdapter` will handle loading the data from there.

### 3. 🧪 Run the Demo

To make sure everything is working correctly, you can run the `test_and_demo.py` script. This will generate a small synthetic dataset and run a quick test of the entire pipeline.

```bash
python test_and_demo.py
```
**Note:** The test script may fail in some environments due to a deep incompatibility between the installed versions of NumPy and other libraries. This is a known issue that could not be resolved in the development environment.

### 4. 🧠 Train the Model

To train the model on a real dataset, you can use the `main.py` script. You will need to specify the name of the dataset you want to use.

```bash
# Example: Train on the Brown2019 dataset
python main.py --dataset_name Brown2019 --batch_size 64 --epochs 50
```

## 📈 Next Steps

*   **Experiment!** Try different hyperparameters in `main.py` to improve the model's accuracy.
*   **Contribute!** This is an open-source project, and contributions are welcome.

Happy coding! 🎉
