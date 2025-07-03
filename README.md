# Autism Spectrum Disorder Detection Using ABIDE

This project provides a full pipeline for detecting Autism Spectrum Disorder (ASD) using the ABIDE dataset, including preprocessing, model training, and a web-based frontend for predictions.

---

## 📁 Directory Structure

```
Autism Spectrum Disorder Using ABIDE/
├── DataSet/                  # Contains ABIDE Preprocess dataset splits
│   ├── dataset_splits_fMRI.zip
│   └── dataset_splits_sMRI.zip
├── Pre-processing-code/      # Scripts for data preprocessing
│   ├── Extract.py
│   ├── Preprocess.py
│   ├── split.py
│   └── filtered_abide_1.csv
├── Model-Training-code/      # Model training and evaluation scripts
│   ├── BackBone01.py
│   ├── BackBone02.py
│   ├── Evl-B1.py
│   ├── Evl-B2.py
│   ├── Evl-Fusion.py
│   ├── Fusion.py
│   └── Complete_Project.ipynb
├── Model Weights/            # Pretrained model weights
│   ├── best_fmri_expert.pth
│   ├── best_fusion_model.pth
│   └── best_smri_expert.pth
├── Web app/ # Web frontend (Flask)
│   ├── app.py
│   ├── static/
│   ├── templates/
│   ├── models/
│   ├── results/
│   └── ...
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repo-url>
cd "Autism Spectrum Disorder Using ABIDE"
```

### 2. Set Up Python Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧹 Data Preprocessing
All preprocessing scripts are in `Pre-processing-code/`.

1. **Extract Features:**
   - Run `Extract.py` to extract Selected Images from the raw ABIDE data.
2. **Preprocess Data:**
   - Run `Preprocess.py` to clean and prepare the data for training.
3. **Split Data:**
   - Use `split.py` to split the data into training, validation, and test sets.

> **Note:** Make sure the dataset files are present in the `DataSet/` directory.

---

## 🏋️ Model Training
All model training and evaluation scripts are in `Model-Training-code/`.

- `BackBone01.py`, `BackBone02.py`: Backbone model architectures.
- `Fusion.py`: Model fusion logic.
- `Evl-B1.py`, `Evl-B2.py`, `Evl-Fusion.py`: Evaluation scripts for different models.
- `Complete_Project.ipynb`: Jupyter notebook for end-to-end workflow.

**To train a model:**
```bash
python Model-Training-code/BackBone01.py
# and
python Model-Training-code/BackBone02.py
# and
python Model-Training-code/Fusion.py
```

> **Tip:** Update the scripts as needed for your data paths and parameters.

---

## 🧠 Model Weights
Pretrained weights are stored in `Model Weights/`. You can use these directly for inference or continue training.

---

## 🌐 Running the Frontend (Flask App)
The web frontend is in `Web app/` and is built with Flask.

1. Navigate to the frontend directory:
   ```bash
   cd Web app/
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser and go to `http://127.0.0.1:5000/`

---

## 📝 Notes & Troubleshooting
- Ensure all required Python packages are installed (see `requirements.txt`).
- Place your dataset files in the correct folders as described above.
- If you encounter errors, check file paths and Python version compatibility.
- For GPU acceleration, ensure you have the correct version of PyTorch installed for your CUDA version.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 🤝 Team
1. Muhammad Qadeer
2. Alyan Umair 
3. Muhammad Shayan
4. Gulraiz Khan 
5. Fiza Abull Razzaq (Supervisor)
