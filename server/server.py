
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import traceback
from testing.data_parallel import get_default_device
from testing.test_video import fake_or_real  # Import the function from test_video.py

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure you have a directory to store uploaded videos
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


device=get_default_device()
print("----------------------------Model Loading----------------------------")
model_1= torch.load(os.path.join("Trained_Models","model_1.pth"), map_location=device)
model_1.to(device)##### Put Model on the CPU/GPU
model_1.eval()##### Start the Evaluation Mode for the Pytorch Model

model_2= torch.load(os.path.join("Trained_Models","model_2.pth"), map_location=device)
model_2.to(device)##### Put Model on the CPU/GPU
model_2.eval()##### Start the Evaluation Mode for the Pytorch Model
print("------------------------Both Models Loaded Correctly------------------------")



@app.route('/')
def home():
    return "Deep Learning Model Server is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        video_file = request.files['file']

        print(video_file)
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Debugging: Confirm that the file object is valid
        print(f"File name: {video_file.filename}")
        # print(f"File size: {len(video_file.read())} bytes")

        # Rewind the file back to the start after reading its size
        video_file.seek(0)

        if len(video_file.read()) == 0:
            print("E1")
            return jsonify({'error': 'File1 is empty'}), 400
        
        video_file.seek(0)

        # Save the file to the server
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
        video_file.save(video_path)

        print(video_path)

        video_file.seek(0)

        if len(video_file.read()) == 0:
            print("E2")
            return jsonify({'error': 'File2 is empty'}), 400

        video_file.seek(0)

        

        # Process the uploaded video using the 'fake_or_real' function
        status_data = fake_or_real(
            model_1=model_1,
            model_2=model_2,
            ensemble_strat=1,
            conf_strat=1,
            video_path=video_path,
            per_frame=10,
            device=device
        )

        print("Processed Status Data:", status_data)

        # Clean up the video file after processing
        if os.path.exists(video_path):
            os.remove(video_path)

        # Return the results as a JSON response
        return jsonify({
            "status": "Fake" if status_data[0] >= 0.5 else "Real",
            "mean_fake_score": status_data[0],
            "total_frames": status_data[1],
            "total_evaluated_frames": status_data[2],
            "total_fake_frames": status_data[3],
            "total_real_frames": status_data[4]
        }), 200

    except Exception as e:
        print("An error occurred:", str(e))
        print("Exception traceback:", traceback.format_exc())
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
