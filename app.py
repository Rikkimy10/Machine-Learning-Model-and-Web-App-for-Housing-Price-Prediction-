from flask import Flask, request, render_template
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load model and columns
with open("model/bangalore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("model/columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

# Prediction function
def predict_price(location, sqft, bath, bhk):
    loc_index = data_columns.index(location.lower()) if location.lower() in data_columns else -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)

# Flask route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        location = request.form["location"]
        sqft = float(request.form["sqft"])
        bath = int(request.form["bath"])
        bhk = int(request.form["bhk"])
        price = predict_price(location, sqft, bath, bhk)
        return render_template("index.html", price=price)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
