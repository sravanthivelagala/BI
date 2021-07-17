import os 
from flask import Flask, request, jsonify, render_template,redirect, url_for
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')

from keras.models import load_model 
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 

global sess
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
set_session(sess)
global model 
model = load_model('model.h5') 
global graph
graph = tf.compat.v1.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('predict', filename=filename))
    return render_template('index.html')


@app.route('/predict/<filename>')
def predict(filename):
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    my_image_re = resize(my_image, (32,32,3))
    
    #Step 3
    with graph.as_default():
      set_session(sess)
      probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
      output = round(probabilities[0], 2)
      print(probabilities)

    if output == 0:
        return render_template('index.html',
                               prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN')
                                
    else:
        return render_template('index.html',
                               prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER')


if __name__ == "__main__":
    app.run(debug=True)