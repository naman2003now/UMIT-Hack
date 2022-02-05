import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
pickle_in = open(r'/mnt/e/main/ml/api/temp/Model.pckl', 'rb')
pickle_model = pickle.load(pickle_in)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    request_data = request.get_json()

    lrank = request_data['lrank']
    hrank = request_data['hrank']

    data = [[lrank, hrank]]
    #data = np.array(data)
    #pred = pickle_model.predict(data.reshape(1,-1))
    pred = pickle_model.predict(data)
    #km2 = pickle.load(pickle_in)    
    return jsonify(pred.tolist())

if __name__ == '__main__':
    app.run(debug=True)
