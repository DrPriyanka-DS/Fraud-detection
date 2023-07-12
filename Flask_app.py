from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model('xgb_model.xgb')

# Load the LabelEncoders
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''
    if request.method == 'POST':
        # Get data from POST request
        data = request.form
        df = pd.DataFrame([data])

        # Convert EVENT_ID to integer
        df['EVENT_ID'] = df['EVENT_ID'].astype(int)

        # Ensure the order of columns matches the input of the trained model
        df = df[['EVENT_ID', 'CHANNEL', 'USER_ID', 'IP_ADDRESS', 'ISP', 'PAYEE_ID']]

        # Transform categorical features
        for column in ['CHANNEL', 'USER_ID', 'IP_ADDRESS', 'ISP', 'PAYEE_ID']:
            if column in df.columns:
                le = label_encoders.get(column)
                if le:
                    df[column] = le.transform(df[column])

        # Make the prediction
        prediction = xgb_clf.predict(df)

        # Map the binary output back to 'Genuine' and 'Fraud'
        if prediction[0] == 0:
            prediction = 'Prediction: Genuine'
        else:
            prediction = 'Prediction: Fraud'

    return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
