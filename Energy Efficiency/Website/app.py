from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('Model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form
    input_features = [float(x) for x in request.form.values()]
    pred = model.predict([input_features])[0]

    return render_template('result.html', prediction_hl=round(pred[0],3), prediction_cl=round(pred[1],3))  # Creates result.html to show predictions


app.run(debug=True)