from flask import Flask, request, render_template, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

from src.logger import logging


application = Flask(__name__)

app = application


@app.route('/')
def home_page():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])  # form submit process...
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')

    else:
        data = CustomData(
            age=float(request.form.get('age')),
            workclass=request.form.get('workclass'),
            education=float(request.form.get('education')),
            occupation=request.form.get('occupation'),
            fnlwgt=request.form.get('fnlwgt'),
            relationship=request.form.get('relationship'),
            capital_gain=float(request.form.get('capital_gain')),
            capital_loss=float(request.form.get('capital_loss')),
            # Type_of_order = Type_of_order,
            hours_per_week=int(request.form.get('hours_per_week'))
        )

        final_new_data = data.get_data_as_dataframe()
        logging.info(
            f"final new data - {type(final_new_data)} and value-{final_new_data}")

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        results = round(pred[0], 2)
        result_str = ""

        if results == 1:
            result_str = "Your income is more than >50k"
        else:
            result_str = "Your income is less than <=50K"

            # return render_template('result.html', final_result=results)
        return render_template('main.html', original_input={'age': final_new_data["age"]}, result=result_str)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    # run at localhost... http://127.0.0.1:5000/
