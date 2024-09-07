from flask import Flask, request, render_template
import pickle
app = Flask(__name__)

with open('spam.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']
    
    transformed_message = vectorizer.transform([message])
    
    prediction = model.predict(transformed_message)
    
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    return result

if __name__ == '__main__':
    app.run(debug=True)