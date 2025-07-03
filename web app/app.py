from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import random
import shutil
from helper import preprocess as prp
from helper import inference as irf

app = Flask(__name__)

# Configure static folder
app.static_folder = 'static'

BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")         

DIRS = {
    "temp_smri" : os.path.join(DATA_DIR, "temp", "smri"),
    "temp_fmri" : os.path.join(DATA_DIR, "temp", "fmri"),
    "smri_slices" : os.path.join(DATA_DIR, "smri_slices"),
    "fmri_slices" : os.path.join(DATA_DIR, "fmri_slices"),  
}

for path in DIRS.values():
    os.makedirs(path, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():

    for keys in DIRS.keys():
        clear_dir_contents(DIRS[keys])
        
    data_type = request.form.get('dataType')
    analysis_type = request.form.get('analysisType')

    print(f"-------- You have selected {data_type} --------")
    print(f"-------- You have selected {analysis_type} --------")
    
    if data_type == 'Fusion':
        # Check if both files are present
        if 'sMRIFile' not in request.files or 'fMRIFile' not in request.files:
            return jsonify({'error': 'Both sMRI and fMRI files are required for Fusion analysis'}), 400
        
        smri_file = request.files['sMRIFile']
        fmri_file = request.files['fMRIFile']
        
        if smri_file.filename == '' or fmri_file.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400
        
        if not smri_file.filename.endswith(('.nii', '.nii.gz')) or not fmri_file.filename.endswith(('.nii', '.nii.gz')):
            return jsonify({'error': 'Files must be .nii or .nii.gz format'}), 400
        
        # Save the files temporarily
        
        smri_file_path = os.path.join(DIRS['temp_smri'], smri_file.filename)
        fmri_file_path = os.path.join(DIRS['temp_fmri'], fmri_file.filename)
        
        smri_file.save(smri_file_path)
        fmri_file.save(fmri_file_path)
        
        # Process the files based on data_type and analysis_type
        prp.preprocess_images(DIRS['temp_smri'] , DIRS['smri_slices'])
        prp.preprocess_images(DIRS['temp_smri'] , DIRS['fmri_slices'])

        result = frontend_output(data_type, analysis_type)
        
        
    else:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.endswith(('.nii', '.nii.gz')):
            return jsonify({'error': 'File must be a .nii or .nii.gz file'}), 400
        

        if data_type == 'sMRI':

            smri_file_path = os.path.join(DIRS['temp_smri'], file.filename)
            file.save(smri_file_path)

            prp.preprocess_images(DIRS['temp_smri'] , DIRS['smri_slices'])

            lbl, conf_sel = irf.predict_scan(DIRS['smri_slices'] , 'models/best_smri_expert.pth')
            print(lbl)
            result = frontend_output(lbl, conf_sel, data_type, analysis_type) 


        else:

            fmri_file_path = os.path.join(DIRS['temp_fmri'], file.filename)
            file.save(fmri_file_path)
            prp.preprocess_images(DIRS['temp_fmri'] , DIRS['fmri_slices']) 

            lbl, conf_sel = irf.predict_scan(DIRS['fmri_slices'] , 'models/best_fmri_expert.pth')
            result = frontend_output(lbl, conf_sel, data_type, analysis_type)
  

    
    return jsonify(result)

def frontend_output(label , confidence, data_type, analysis_type):

    if label == 0:
        is_asd = "ASD Positive"

    else:
        is_asd = 'ASD Negative'
    
    result_image = r'static\images\brain-transparent.png'

    
    return {
        'prediction': is_asd,
        'confidence': round(confidence * 100, 3),
        'model_used': f"{data_type} {analysis_type}",
        'result_image': result_image
    }

def process_fusion_with_ml_model(smri_file_path, fmri_file_path, analysis_type):
    """
    Process the sMRI and fMRI files with the fusion ML model.
    
    In a real application, this function would:
    1. Load the fusion ML model
    2. Preprocess both .nii files
    3. Run the model on the preprocessed data
    4. Return the results
    
    For this demo, we'll simulate processing time and return mock results.
    """
    # Simulate processing time
    time.sleep(3)
    
    # Select the appropriate model
    model_path = 'models/model.pth02'
    
    # Mock results
    confidence = random.uniform(0.90, 0.98)  # Higher confidence for fusion
    is_asd = confidence > 0.5
    
    # Generate result image path
    result_image = 'static/images/result-fusion.jpg'
    
    return {
        'prediction': 'ASD Positive' if is_asd else 'ASD Negative',
        'confidence': round(confidence * 100, 1),
        'model_used': f"Fusion {analysis_type} ({os.path.basename(model_path)})",
        'result_image': result_image,
        'fusion_details': {
            'sMRI_contribution': f"{random.randint(40, 60)}%",
            'fMRI_contribution': f"{random.randint(40, 60)}%"
        }
    }

@app.route('/download/<filename>')
def download_result(filename):
    
    return send_from_directory('results', filename)

@app.route('/contact', methods=['POST'])
def contact():
    
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    
    # Process the contact form data
    # For demo purposes, we'll just return a success message
    return jsonify({'success': True, 'message': f'Thank you, {name}! Your message has been received.'})


def clear_dir_contents(path: str):
    
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            shutil.rmtree(full)
        else:
            os.remove(full)

    print(f"{path} is cleaned")

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # In a real application, you would load your ML models here
    # For example:
    # smri_model = load_model('models/model.pth')
    # fmri_model = load_model('models/model.pth01')
    # fusion_model = load_model('models/model.pth02')
    
    app.run(debug=True)
