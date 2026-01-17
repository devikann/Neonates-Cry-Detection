from flask import Flask, render_template
import predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    result = predict.run_detection()
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
