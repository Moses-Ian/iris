from flask import Flask,render_template,session,url_for,redirect
from wtforms import TextField,SubmitField
from flask_wtf import FlaskForm
import numpy as np
import joblib
import tensorflow

def return_prediction(model,scaler,sample_json):
  # pull out the data
  s_len = sample_json["sepal_length"]
  s_wid = sample_json["sepal_width"]
  p_len = sample_json["petal_length"]
  p_wid = sample_json["petal_width"]

  # shape and scale the data
  flower = [[s_len, s_wid, p_len, p_wid]]
  flower = scaler.transform(flower)

  # make a prediction
  predict_x = model.predict(flower) 
  class_index = np.argmax(predict_x,axis=1)
  classes = np.array(['setosa', 'versicolor', 'virginica'])
  return classes[class_index]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class FlowerForm(FlaskForm):
    sep_len = TextField("Sepal Length")
    sep_wid = TextField("Sepal Width")
    pet_len = TextField("Petal Length")
    pet_wid = TextField("Petal Width")
    submit = SubmitField("Analyze")

# home page
@app.route("/",methods=['GET','POST'])
def index():
    form = FlowerForm()
    
    # if it's a submission, redirect to the prediction page
    if form.is_submitted() and form.validate():
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data
        return redirect(url_for("prediction"))
        
    # if it's normal, show them the home page
    return render_template('home.html',form=form)

flower_model = tensorflow.keras.models.load_model('final_iris_model.h5', compile=False)
flower_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
flower_scaler = joblib.load('iris_scaler.pkl')

# prediction page
@app.route('/prediction')
def prediction():
    # parse the form inputs
    content = {}
    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] =  float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] =  float(session['pet_wid'])
    
    # generate the prediction
    results = return_prediction(flower_model,flower_scaler,content)
    
    # return a page with the prediction displayed
    return render_template('prediction.html',results=results)

if __name__=='__main__':
    app.run()