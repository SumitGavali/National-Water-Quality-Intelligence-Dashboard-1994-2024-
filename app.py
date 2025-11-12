# app.py - Flask Backend API
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load trained model and preprocessing objects
try:
    model = joblib.load('water_potability_predictor.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("✅ Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    feature_names = []

def create_advanced_features(input_data):
    """Recreate the same feature engineering as in training"""
    df = pd.DataFrame([input_data])
    
    # Advanced feature engineering (same as training)
    df['TDS_Index'] = df['Solids'] / 1000
    df['Hardness_Risk'] = np.where(df['Hardness'] > 200, 1, 0)
    df['pH_Risk'] = np.where((df['ph'] < 6.5) | (df['ph'] > 8.5), 1, 0)
    df['Turbidity_Risk'] = np.where(df['Turbidity'] > 5, 1, 0)
    df['pH_Hardness_Interaction'] = df['ph'] * df['Hardness']
    df['Organic_Trihalomethanes'] = df['Organic_carbon'] * df['Trihalomethanes']
    df['Chemical_Score'] = (df['Chloramines'] + df['Sulfate'] + df['Trihalomethanes']) / 3
    df['Physical_Score'] = (df['Hardness'] + df['Solids'] + df['Turbidity']) / 3
    
    # Domain-informed features
    df['WQI_Simple'] = (df['ph']/8.5 + df['Hardness']/200 + 
                       (14-df['Chloramines'])/14 + df['Sulfate']/250) / 4
    df['pH_Deviation'] = abs(df['ph'] - 7.0)
    
    return df

@app.route('/')
def home():
    return jsonify({
        "message": "Water Quality Prediction API",
        "status": "active",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(feature_names)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required parameters
        required_params = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]
        
        for param in required_params:
            if param not in data:
                return jsonify({
                    "error": f"Missing parameter: {param}",
                    "required_parameters": required_params
                }), 400
        
        # Create feature engineered input
        engineered_data = create_advanced_features(data)
        
        # Ensure correct feature order
        engineered_data = engineered_data[feature_names]
        
        # Scale features
        scaled_data = scaler.transform(engineered_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0]
        
        # Interpret results
        potability_status = "POTABLE" if prediction == 1 else "NOT POTABLE"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        # Feature importance explanation
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            top_features_explanation = [
                f"{feat} ({imp:.1%})" for feat, imp in top_features
            ]
        else:
            top_features_explanation = ["Feature importance not available"]
        
        # Water quality insights
        insights = generate_water_quality_insights(data)
        
        return jsonify({
            "prediction": int(prediction),
            "status": potability_status,
            "confidence": float(confidence),
            "probabilities": {
                "potable": float(probability[1]),
                "not_potable": float(probability[0])
            },
            "top_influencing_factors": top_features_explanation,
            "quality_insights": insights,
            "input_parameters": data
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

def generate_water_quality_insights(data):
    """Generate human-readable insights about water quality"""
    insights = []
    
    # pH analysis
    if data['ph'] < 6.5:
        insights.append("⚠️ Low pH (acidic) may indicate corrosion or metal leaching")
    elif data['ph'] > 8.5:
        insights.append("⚠️ High pH (alkaline) may affect disinfection efficiency")
    else:
        insights.append("✅ pH within optimal range (6.5-8.5)")
    
    # Chloramines analysis
    if data['Chloramines'] > 4:
        insights.append("⚠️ High chloramine levels may cause taste/odor issues")
    else:
        insights.append("✅ Chloramines within acceptable limits")
    
    # Hardness analysis
    if data['Hardness'] > 200:
        insights.append("💧 Hard water detected - may cause scaling")
    else:
        insights.append("✅ Water hardness at acceptable levels")
    
    # Turbidity analysis
    if data['Turbidity'] > 5:
        insights.append("🌫️ High turbidity - potential microbial protection concern")
    else:
        insights.append("✅ Good water clarity")
    
    return insights

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch predictions"""
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        results = []
        for sample in samples:
            engineered_data = create_advanced_features(sample)
            engineered_data = engineered_data[feature_names]
            scaled_data = scaler.transform(engineered_data)
            
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
            
            results.append({
                "prediction": int(prediction),
                "confidence": float(probability[1] if prediction == 1 else probability[0]),
                "status": "POTABLE" if prediction == 1 else "NOT POTABLE"
            })
        
        return jsonify({
            "batch_id": f"batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            "results": results,
            "total_samples": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)