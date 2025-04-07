from flask import Flask, request, jsonify
from flask_cors import CORS
from regression_model import LinearRegressionModel
import os
import time

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:3000"
]}})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "CORS is working!"})

@app.route('/api/get-image', methods=['POST'])
def generate_image():
    global id
    data = request.get_json()
    learning_rate = data.get('learningRate', 0.01)
    epochs = data.get('epochs', 500)

    folder = os.path.join(app.root_path, 'static')
    for filename in os.listdir(folder):
        if filename.startswith("generated_plot") and filename.endswith(".png"):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    model = LinearRegressionModel(learning_rate=learning_rate, epochs=epochs)
    image_path, predictions_df, mse = model.visualize_data()

    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404


    host_url = os.environ.get('HOST_URL', request.host_url)
    image_url = host_url + f"static/{os.path.basename(image_path)}?t={int(time.time())}" # Add timestamp to avoid caching issues and make images unique


    predictions = predictions_df[['# Date', 'Predicted_Receipt_Count']].copy()
    predictions['# Date'] = predictions['# Date'].astype(str)
    predictions_list = predictions.to_dict(orient='records')

    return jsonify({
        "imageUrl": image_url,
        "predictions": predictions_list,
        "mse": mse,
    })

if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
