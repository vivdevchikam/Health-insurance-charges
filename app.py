
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

data = pd.read_csv('health_insurance.csv')

model = joblib.load('RandomForestRegressor.joblib')



@app.route('/')
def index():
  # Gender = 0
  gender = sorted(data['sex'].unique())
  gender.insert(0, ' ')
  region = sorted(data['region'].unique())
  region.insert(0, "Region")
  smoker = sorted(data['smoker'].unique())
  smoker.insert(0, "Smoker")
  return render_template('index.html', genders=gender, regions=region, smokers=smoker)


@app.route('/predict', methods=['POST'])
def predict():
  gender = request.form.get('gender')
  bmi = request.form.get('bmi')
  children = request.form.get('children')
  smoker = request.form.get('smoker')
  region = request.form.get('region')
  age = request.form.get('age')
  print(gender, bmi, children, smoker, region, age)
  input = pd.DataFrame([[age, gender, bmi, children, smoker, region]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
  prediction = model.predict(input)[0]
  return str(np.round(prediction, 2))



if __name__ == "__main__":
  app.run(debug=True, host="localhost", port=5002
)