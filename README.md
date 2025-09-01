# 🚀 GlucoGuard: A Deep Learning Pipeline for Glucose Prediction 🩸

## 🌟 For Patients and Families: A Weather Forecast for Your Glucose

Living with diabetes means constantly thinking about your glucose levels. It can be like trying to predict the weather – sometimes you get it right, and sometimes you're caught in a storm.

**GlucoGuard is like a smart weather forecast for your glucose levels.**

Instead of just telling you what your glucose is *right now*, it looks at your past glucose patterns, your heart rate, your activity, and other factors to predict what your glucose will be in the near future.

**How does this help?**
*   **Early Warnings:** Imagine getting an alert on your phone or watch that says, "In 30 minutes, your glucose is likely to be low." This gives you time to have a snack *before* you start feeling the effects of a hypo.
*   **Peace of Mind:** Especially at night, it can be worrying not knowing what your glucose levels are doing. GlucoGuard can watch over you and alert you or a family member if it predicts a dangerous change.
*   **Better Control:** By understanding how your body reacts to food, exercise, and sleep, GlucoGuard can help you make small changes to your daily routine that can lead to better overall glucose control and a healthier, happier life.

This project is the first step towards building a "digital twin" – a computer model of your body that can help you and your doctor make the best decisions for your health.

---

## 🔬 For Health Practitioners: Proactive Glycemic Control with Predictive Analytics

This project provides a robust, research-grade pipeline for analyzing continuous glucose monitoring (CGM) data and developing predictive models for proactive glycemic control.

**Clinical Significance:**
The management of diabetes is shifting from reactive to proactive care. This pipeline leverages state-of-the-art machine learning to forecast glycemic excursions, enabling timely interventions that can reduce the risk of acute complications (hypo/hyperglycemia) and potentially long-term micro- and macrovascular complications.

**Methodology:**
The core of this project is a transformer-based deep learning model (`HierarchicalGlucosePredictor`) that is trained on time-series data from CGM sensors. The model's architecture is designed to capture complex temporal dependencies in glucose dynamics.

**Feature Engineering:**
The data pipeline automatically calculates a comprehensive suite of over 100 features, including:
*   **Standard Statistical Features:** Rolling means, standard deviations, and coefficients of variation over multiple time windows.
*   **Rate of Change (ROC) Dynamics:** First and second derivatives of the glucose signal to capture velocity and acceleration.
*   **Clinically Validated Glycemic Variability Metrics:** The pipeline uses a custom-built, clinically-validated feature engineering module to calculate key metrics such as:
    *   **MAGE (Mean Amplitude of Glycemic Excursions):** Measures significant glucose swings.
    *   **MODD (Mean of Daily Differences):** Assesses day-to-day glycemic variability.
    *   **CONGA (Continuous Overall Net Glycemic Action):** Evaluates variability over n-hour periods.
    *   **LBGI/HBGI (Low/High Blood Glucose Index):** Quantifies the risk of hypo- and hyperglycemia.
    *   **GRADE:** A comprehensive score for glycemic control.
*   **Signal Processing Features:** FFT and wavelet transforms to capture periodic and non-stationary patterns in the glucose signal.
*   **Optional Wearable Data Fusion:** The pipeline can automatically detect and incorporate data from wearable sensors (e.g., heart rate, HRV, skin temperature). If present, rolling statistics are calculated for these features and they are fused with the CGM data inside the model to provide additional context for predictions.

**Potential Applications:**
*   **Research:** This pipeline can be used to analyze large CGM datasets (like the Awesome-CGM datasets it is designed for) to investigate the effects of different interventions on glycemic control.
*   **Clinical Decision Support:** A trained model could be integrated into a clinical dashboard to provide physicians with predictive insights into their patients' glycemic control.
*   **Personalized Diabetes Management:** The model can be fine-tuned on individual patient data to provide personalized predictions and alerts.

---

## 💻 For Developers: Technical Details

### 📁 File Structure

*   **`main.py`**: The main entry point for running the training pipeline. 🏃‍♂️
*   **`test_and_demo.py`**: A script to generate synthetic data and run a smoke test of the entire pipeline. 🧪
*   **`requirements.txt`**: A list of all the Python packages needed to run the project. 📦
*   **`src/`**: The main source code directory.
    *   **`data_processing/`**: Modules for handling data loading and preprocessing.
        *   **`adapter.py`**: The `AwesomeCGMAdapter` class.
        *   **`cgm_dataset.py`**: The `AdvancedCGMDataset` class.
    *   **`models/`**: The deep learning model architecture.
        *   **`building_blocks.py`**: `MultiHeadTemporalAttention` and `GatedResidualNetwork`.
        *   **`predictor.py`**: The main `HierarchicalGlucosePredictor` model.
    *   **`training/`**: Modules for training the model.
        *   **`losses.py`**: The custom `GlucosePredictionLoss` function.
        *   **`lightning_module.py`**: The `GlucoseLightningModule`.
    *   **`inference/`**: Modules for running the model in a real-time setting.
        *   **`monitor.py`**: `RealTimeGlucoseMonitor` and `AlertManager`.
        *   **`postprocessing.py`**: `EnsemblePredictor` and `PostProcessor`.

### 🚀 How to Use

#### 1. 📦 Installation
It is highly recommended to use a virtual environment.
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (if you have an NVIDIA GPU)
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install the rest of the packages
pip install -r requirements.txt
```

#### 2. 💾 Download the Data
This pipeline is designed to work with the [Awesome-CGM datasets](https://github.com/IrinaStatsLab/Awesome-CGM). Download the datasets and place them in the `cgm_data` directory.

#### 3. 🧪 Run the Demo
The `test_and_demo.py` script will generate synthetic data and run a quick test of the pipeline.
```bash
python test_and_demo.py
```

#### 4. 🧠 Train the Model
Use the `main.py` script to train the model on a real dataset.
```bash
# Example: Train on the Brown2019 dataset
python main.py --dataset_name Brown2019 --batch_size 64 --epochs 50
```

#### 5.  Tune the Model (Optional)
The `optimize.py` script uses the Optuna library to perform hyperparameter optimization, finding the best combination of settings for the model on a given dataset.

```bash
# Run the optimization for 25 trials
python optimize.py
```
The script will output the best validation loss and the corresponding set of hyperparameters. You can then use these values to configure the model for final training in `main.py`.
