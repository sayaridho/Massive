from flask import Flask, render_template,request,redirect
from werkzeug.utils import secure_filename #module yang akan digunakan untuk upload file
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

Upload_Folder = 'uploads' # Path folder untuk menyimpan images yang diupload
app.config['Upload_Folder'] = Upload_Folder
Ext = {'png','jpg','jpeg'} # Format extension yang diterima

model = load_model('models/CNN.keras', compile=False)

def allowed_file(filename): # Fungsi untuk memeriksa apakah file memiliki ekstensi yang diizinkan
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Ext 

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(128, 128))  # Adjust the target size to match your model's expected input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalizing the image if needed
    return image

@app.route('/', methods=['GET','POST'])
def Detect():
    predicted_class = None
    if request.method == "POST":
        # Memastikan apakah file telah diupload
        if 'Foto' not in request.files: 
            return 'No file part', 404
        file = request.files['Foto']
        if file.filename == '':
            return 'No selected file' , 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['Upload_Folder'], filename)
            file.save(filepath) #Save file pada path

            image = preprocess_image(filepath)
            predict = model.predict(image)

            class_labels = ['Hama terdeteksi', 'Hama tidak terdeteksi']
            predicted_class = class_labels[np.argmax(predict)]
            
        else:
            return 'File not allowed', 400   
    return render_template('Detect.html', predicted_class=predicted_class)
    
if __name__ == '__main__':
    app.run(debug=True)