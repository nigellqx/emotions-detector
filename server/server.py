from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/classify_emotion', methods=['GET', 'POST'])
def classify_emotion():
    image_data = request.form['image_data']
    response = jsonify(util.classify_emotion(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(port=5000)