from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import json
import os

app = Flask(__name__)

# Global variables to store model and data
model = None
cars_data = None
model_stats = {}

def load_and_train_model():
    """Load data and train the regression model"""
    global model, cars_data, model_stats
    
    # Load the data
    cars_data = pd.read_excel("cars.xlsx")
    
    # Remove influential observation (index 76)
    cars_new = cars_data.drop(cars_data.index[[76]]) if len(cars_data) > 76 else cars_data
    
    # Train the final model (without WT due to high VIF)
    model = smf.ols('MPG ~ VOL + SP + HP', data=cars_new).fit()
    
    # Calculate statistics
    model_stats = {
        'r_squared': round(model.rsquared, 4),
        'adj_r_squared': round(model.rsquared_adj, 4),
        'f_statistic': round(model.fvalue, 2),
        'aic': round(model.aic, 2),
        'bic': round(model.bic, 2)
    }
    
    # Get coefficient information
    model_stats['coefficients'] = {
        'Intercept': round(model.params['Intercept'], 4),
        'VOL': round(model.params['VOL'], 4),
        'SP': round(model.params['SP'], 4),
        'HP': round(model.params['HP'], 4)
    }
    
    return model

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        # Extract input values
        hp = float(data['hp'])
        sp = float(data['sp'])
        vol = float(data['vol'])
        
        # Create prediction dataframe
        input_data = pd.DataFrame({
            'HP': [hp],
            'SP': [sp],
            'VOL': [vol]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'inputs': {
                'HP': hp,
                'SP': sp,
                'VOL': vol
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model-stats')
def get_model_stats():
    """Return model statistics"""
    return jsonify(model_stats)

@app.route('/data-summary')
def data_summary():
    """Return summary statistics of the dataset"""
    try:
        summary = cars_data[['HP', 'SP', 'VOL', 'MPG']].describe().to_dict()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Load and train model on startup
    print("Loading data and training model...")
    load_and_train_model()
    print("Model trained successfully!")
    print(f"R-squared: {model_stats['r_squared']}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8000)