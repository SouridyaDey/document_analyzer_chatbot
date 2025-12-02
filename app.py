from flask import Flask, render_template, request, jsonify
from rag_pipeline import ask_question

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods = ['POST'])
def chat():
    user_input = request.form['message']
    response = ask_question(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug = True)
