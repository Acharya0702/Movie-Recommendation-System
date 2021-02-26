import numpy as np
from flask import Flask, request, jsonify, render_template
import sklearn.externals
import joblib

app = Flask(__name__)
model = joblib.load(open('Diabetes.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        feature_array = [f1,f2,f3,f4]
        feature = np.array(feature_array).reshape(1,-1)
        prediction = model.predict(feature)
        dic = {'No, you dont have diabetes.':0, 'Yes, You have Diabetes.':1}
        
        for key,value in dic.items():
            if value == prediction:
                x=key
        return render_template('index.html', prediction='Prediction {}'.format(x))

if __name__ == '__main__':
    app.run(debug=True)