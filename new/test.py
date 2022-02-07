import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sklearn
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np

ranks = []
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class Model:
  i = '0'
  lrank = 1
  hrank = 2
  r_names = None
  r_ID = None
  corr_ID = None
  recc = None
  cmat = []
  fir = []
  sec = []

  def predict(self):
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(self.cmat)
    df=pd.DataFrame(decomposed_matrix)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    product_names = list(self.cmat.index)
    product_ID = product_names.index(str(self.i))
    correlation_product_ID = correlation_matrix[product_ID]
    Recommend = list(self.cmat.index[correlation_product_ID > 0.95])
    fl = []
    for i in range(len(correlation_product_ID)):
      if(correlation_product_ID[i] > 0.95):
        fl.append([correlation_product_ID[i], self.cmat.index[i]])
    fl.sort(reverse=True)
    flf = []
    for i in range(len(fl)):
      if (fl[i][0] > 0.95):
        flf.append(fl[i])

    for i in flf:
      for j in self.cmat.loc[i[1]].items():
        if(j[1] == 5 and j[0] not in self.fir and (ranks[j[0]][1] > self.lrank)):
          self.sec.append([(j[0].split())[0], j[0][len((j[0].split())[0]):], ranks[j[0]][0], ranks[j[0]][1]])
        elif (j[1] == 2 and j[0] not in self.sec and (ranks[j[0]][1] > self.lrank)):
          self.sec.append([(j[0].split())[0], j[0][len((j[0].split())[0]):], ranks[j[0]][0], ranks[j[0]][1]])

    print(self.fir, self.sec, sep = '\n\n')
    final = []
    for i in self.fir:
        final.append(i)
    for i in self.sec:
        final.append(i)
    finalstr = str()
    for i in final:
        finalstr += str(i) + '\n'
    return jsonify(final)

@app.route('/')
def hello():
  return 'hi'

@app.route('/predict', methods=['POST']) #or POST u see that
@cross_origin()
def predict():
  #take all these as input from args
  global ranks
  req_dat = request.get_json()
  lrank = req_dat['lrank']#5000
  hrank = req_dat['hrank']#7000
  stream1 = req_dat['stream1']#'Computer Science'
  stream2 = req_dat['stream2']#'Electronics'
  
  f = open('essentials.pckl', 'rb')
  f1 = pickle.load(f)
  f.close()
  #print(f1)

  f = open('Model2.pckl', 'rb')
  f2 = pickle.load(f)
  f.close()

  wsc = f1[0]
  ranks = f1[1]
  f2.cmat = f1[2]
  
  f2.lrank = lrank
  f2.hrank = hrank
  f2.stream1 = stream1
  f2.stream2 = stream2
  f2.i = wsc[(stream1, stream2)]
  return f2.predict()

if __name__ == '__main__':
  app.run(debug = True)
