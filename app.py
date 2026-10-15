import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the Linear Regression model from the pickle file[cite: 1]
with open('linear.pkl', 'rb') as f:
  model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-gradient: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            --card-bg: #ffffff;
            --text-color: #1f2937;
        }
        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background: var(--bg-gradient);
            color: var(--text-color);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 600px;
        }
        h1 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 8px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 13px;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 15px;
            box-sizing: border-box;
            transition: all 0.2s ease;
        }
        input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
            outline: none;
        }
        button {
            width: 100%;
            padding: 14px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #e0e7ff;
            border-left: 6px solid var(--primary);
            border-radius: 8px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #312e81;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Property Valuation</h1>
        <div class="subtitle">Enter property details below to estimate market value</div>
        
        <form method="POST">
            <div class="grid">
                <div class="form-group">
                    <label for="Square_Footage">Square Footage</label>
                    <input type="number" step="any" id="Square_Footage" name="Square_Footage" placeholder="e.g. 2000" required>
                </div>
                <div class="form-group">
                    <label for="Num_Bedrooms">Bedrooms</label>
                    <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" placeholder="e.g. 3" required>
                </div>
                <div class="form-group">
                    <label for="Num_Bathrooms">Bathrooms</label>
                    <input type="number" step="any" id="Num_Bathrooms" name="Num_Bathrooms" placeholder="e.g. 2" required>
                </div>
                <div class="form-group">
                    <label for="Year_Built">Year Built</label>
                    <input type="number" id="Year_Built" name="Year_Built" placeholder="e.g. 2010" required>
                </div>
                <div class="form-group">
                    <label for="Lot_Size">Lot Size</label>
                    <input type="number" step="any" id="Lot_Size" name="Lot_Size" placeholder="e.g. 5000" required>
                </div>
                <div class="form-group">
                    <label for="Garage_Size">Garage Size</label>
                    <input type="number" step="any" id="Garage_Size" name="Garage_Size" placeholder="e.g. 2" required>
                </div>
                <div class="form-group full-width">
                    <label for="Neighborhood_Quality">Neighborhood Quality Score</label>
                    <input type="number" step="any" id="Neighborhood_Quality" name="Neighborhood_Quality" placeholder="e.g. 8" required>
                </div>
            </div>
            <button type="submit">Calculate Prediction</button>
        </form>

        {% if prediction is not none %}
        <div class="result">
            Estimated Value: ${{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def predict():
  prediction = None
  if request.method == "POST":
    try:
      # Map form requests to exact features expected by the pickle file[cite: 1]
      features = [
          float(request.form["Square_Footage"]),
          float(request.form["Num_Bedrooms"]),
          float(request.form["Num_Bathrooms"]),
          float(request.form["Year_Built"]),
          float(request.form["Lot_Size"]),
          float(request.form["Garage_Size"]),
          float(request.form["Neighborhood_Quality"]),
      ]
      final_features = np.array([features])
      pred = model.predict(final_features)[0]
      prediction = f"{pred:,.2f}"
    except Exception as e:
      prediction = f"Error: {str(e)}"

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
