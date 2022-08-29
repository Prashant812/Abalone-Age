import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('recommend.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])


    return render_template('recommend.html',
                           Length = int_features[0],
                           Diameter = int_features[1],
                           Height = int_features[2],
                           Whole_weight = int_features[3],
                           Shucked_weight = int_features[4],
                           Viscera_weight = int_features[5],
                           Shell_weight = int_features[6],
                           prediction_text='Abalone age should be  {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)