from flask import Flask, render_template, request
import pickle

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        email_text = request.form['email']
        result = model.predict([email_text])[0]
        prediction = "Spam" if result == 1 else "Not Spam"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
