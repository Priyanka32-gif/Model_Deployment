from flask import Flask, render_template,request
from Clustering_app import predict_cluster
from Classification_app import predict_classification
import pandas as pd
app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/display_data', methods=['GET'])
def display_data():
    each_position_duration = pd.read_excel('Each_position_duration.xlsx')
    each_position_duration = each_position_duration.to_dict(orient='records')

    total_pick_placed = pd.read_excel('Total_pick_placed.xlsx')
    total_pick_placed = total_pick_placed.to_dict(orient='records')

    return render_template('display_data.html', each_position_duration=each_position_duration, total_pick_placed=total_pick_placed)


@app.route('/predict', methods=['POST'])
def predict():
    data_str = request.form['data1']
    data = [float(x.strip()) for x in data_str.split(',')]  # Split string by comma and convert each part to float

    submit_type = request.form['submit_type']

    if submit_type == 'predict_cluster':
        cluster_label, most_common_target, explanation = predict_cluster(data)
        result = f"Cluster Label: {cluster_label}, Most Common Target: {most_common_target}"
    elif submit_type == 'predict_classification':
        predicted_label, explanation = predict_classification(data)
        result = f"Predicted Label: {predicted_label}"


    return render_template('index.html', result=result, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
