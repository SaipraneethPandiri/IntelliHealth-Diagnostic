from flask import Flask, render_template, request
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')

@app.route('/brain_tumor')
def brain_tumor():
    return render_template('brain_tumor.html')

@app.route('/cancer')
def cancer():
    return render_template('lung_cancer.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dpf')
def dpf():
    return render_template('dpf-calculator.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/bmi')
def bmi():
    return render_template('bmi-calculator.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

with open('diabetes_pred.sav', 'rb') as file:
    db_model = pickle.load(file)

@app.route('/db',methods=['GET','POST'])
def db():

    preg=request.form['Pregnancies']
    glu=request.form['Glucose']
    bp=request.form['BloodPressure']
    st=request.form['SkinThickness']
    ins=request.form['Insulin']
    Bmi=request.form['BMI']
    dpf=request.form['DiabetesPedigreeFunction']
    Age=request.form['Age']

    arr=np.array([preg,glu,bp,st,ins,Bmi,dpf,Age]).reshape(1,-1)
    
    pred=db_model.predict(arr)
    if pred[0] == 1:
        x="Our model predicts that this is a positive case for Diabetes."
        return render_template('diabetes.html',title=x)

    else:
        x="Our model predicts that this is a negative case for Diabetes."
        return render_template('diabetes.html',title=x)
    
with open('heart_disease_random_forest.sav', 'rb') as file:
    hd_model = pickle.load(file)
    
@app.route('/hd',methods=['GET','POST'])
def hd():
    age=request.form['age']
    sex=request.form['sex']
    cp=request.form['chestpain']
    rbp=request.form['restingbp']
    cl=request.form['chol']
    fbs=request.form['fastingbs']
    rcg=request.form['rcg']
    mhr=request.form['maxhr']
    ea=request.form['ea']
    op=request.form['oldpeak']
    slope=request.form['slope']

    

    arr1=np.array([[age,sex,cp,rbp,cl,fbs,rcg,mhr,ea,op,slope]])
    pred=hd_model.predict(arr1)
    x=""
    if pred[0] == 1:
      x="Our model predicts that this is a positive case for heart disease"
    else:
      x="Our model predicts that this is a Negative case for heart disease."
    return render_template('heart_disease.html',title=x)


with open('lc_final.pkl', 'rb') as file:
    lc_model = pickle.load(file)
    
@app.route('/lc',methods=['GET','POST'])
def lc():
    gen=request.form['gender']
    age=request.form['age']
    smo=request.form['smoking']
    yf=request.form['yellowfing']
    anx=request.form['anxiety']
    pp=request.form['pp']
    chr=request.form['chronic']
    fati=request.form['fatigue']
    algy=request.form['allergy']
    whe=request.form['wheezing']
    alc=request.form['alcohol']
    coug=request.form['coughing']
    short=request.form['shortness']
    swal=request.form['swalloing']
    cp=request.form['chestpain']

    arr1=np.array([[gen,age,smo,yf,anx,pp,chr,fati,algy,whe,alc,coug,short,swal,cp]])
    pred=lc_model.predict(arr1)
    x=""
    if pred[0] >= 0.5:
      x="Our model predicts that this is a positive case for Lung Cancer."
    else:
      x="Our model predicts that this is a negative case for Lung Cancer"

    return render_template('lung_cancer.html',title=x)

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
import cv2
import imutils



def crop_brain_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    return new_image

model = load_model('brain_tumor_final_model.h5')

@app.route('/bt',methods=['POST'])
def bt():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        img_pil = Image.open(file).convert('RGB')
        img = np.array(img_pil)

        img = crop_brain_contour(img)

        img = cv2.resize(img, (240, 240))

        img_array = np.array(img, dtype=np.float32)  
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  
        prediction = model.predict(img_array)
        predicted_class = np.round(prediction[0])
        x=""

        if predicted_class == 1:
            x="The model predicts this is a positive case for brain tumor.."
        else:
            x="The model predicts that this is a negative case for brain tumor"

        return render_template('brain_tumor.html',title=x)


if __name__ == "__main__":
    app.run(debug=True)

