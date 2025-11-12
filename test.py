
from flask import Flask, request, jsonify ,send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Debug information
print("🟡 Starting Flask server...")
print("🟡 Current directory:", os.getcwd())
print("🟡 Files in directory:", [f for f in os.listdir('.') if f.endswith('.pkl')])

# Load trained model and preprocessing objects
model = None
scaler = None
feature_names = []
model_loaded = False

def load_models():
    """Load model files with multiple fallback options"""
    global model, scaler, feature_names, model_loaded
    
    model_files = [
        'water_potability_predictor.pkl',
        'water_potability.predictor.pkl',
        'model.pkl'
    ]
    
    scaler_files = [
        'feature_scaler.pkl',
        'scaler.pkl'
    ]
    
    feature_files = [
        'feature_names.pkl',
        'features.pkl'
    ]
    
    # Try to load model
    for model_file in model_files:
        try:
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                print(f"✅ Model loaded from: {model_file}")
                break
        except Exception as e:
            print(f"❌ Failed to load model from {model_file}: {e}")
    
    # Try to load scaler
    for scaler_file in scaler_files:
        try:
            if os.path.exists(scaler_file):
                scaler = joblib.load(scaler_file)
                print(f"✅ Scaler loaded from: {scaler_file}")
                break
        except Exception as e:
            print(f"❌ Failed to load scaler from {scaler_file}: {e}")
    
    # Try to load feature names
    for feature_file in feature_files:
        try:
            if os.path.exists(feature_file):
                with open(feature_file, 'rb') as f:
                    feature_names = pickle.load(f)
                print(f"✅ Feature names loaded from: {feature_file}")
                print(f"✅ Number of features: {len(feature_names)}")
                break
        except Exception as e:
            print(f"❌ Failed to load features from {feature_file}: {e}")
    
    # Check if all components loaded successfully
    if model is not None and scaler is not None and feature_names:
        model_loaded = True
        print("🎉 All model components loaded successfully!")
        print(f"📊 Model type: {type(model)}")
        if hasattr(model, 'feature_importances_'):
            print(f"📈 Model has feature importances")
    else:
        print("⚠️  Running in DEMO MODE - some components not loaded")
        print(f"   Model: {'Loaded' if model else 'Missing'}")
        print(f"   Scaler: {'Loaded' if scaler else 'Missing'}")
        print(f"   Features: {'Loaded' if feature_names else 'Missing'}")

# Load models on startup
load_models()

def create_advanced_features(input_data):
    """Recreate the same feature engineering as in training"""
    try:
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
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return pd.DataFrame([input_data])

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "features_loaded": len(feature_names),
        "model_type": str(type(model).__name__) if model else "None",
        "service": "Water Quality Prediction API"
    })

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Endpoint to reload models without restarting server"""
    global model, scaler, feature_names, model_loaded
    try:
        load_models()
        return jsonify({
            "message": "Models reloaded successfully",
            "model_loaded": model_loaded,
            "features_count": len(feature_names)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to reload models: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "example_format": {
                    "ph": 7.1, "Hardness": 150, "Solids": 25000,
                    "Chloramines": 3.5, "Sulfate": 200, "Conductivity": 400,
                    "Organic_carbon": 2.5, "Trihalomethanes": 65, "Turbidity": 3.2
                }
            }), 400
        
        # Validate required parameters
        required_params = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]
        
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return jsonify({
                "error": f"Missing parameters: {', '.join(missing_params)}",
                "required_parameters": required_params
            }), 400
        
        # Demo mode if model not loaded
        if not model_loaded:
            return demo_prediction(data)
        
        # Create feature engineered input
        engineered_data = create_advanced_features(data)
        
        # Ensure correct feature order
        if not all(feature in engineered_data.columns for feature in feature_names):
            missing_features = [f for f in feature_names if f not in engineered_data.columns]
            return jsonify({
                "error": f"Feature mismatch. Missing: {missing_features}",
                "available_features": list(engineered_data.columns)
            }), 500
        
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
            "input_parameters": data,
            "model_used": "Real ML Model"
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "model_loaded": model_loaded
        }), 500

def demo_prediction(data):
    """Provide demo predictions when model is not loaded"""
    # Simple rule-based demo prediction
    risk_factors = 0
    
    if data['ph'] < 6.5 or data['ph'] > 8.5:
        risk_factors += 1
    if data['Chloramines'] > 4:
        risk_factors += 1
    if data['Turbidity'] > 5:
        risk_factors += 1
    if data['Trihalomethanes'] > 80:
        risk_factors += 1
    
    # Simple demo logic
    is_potable = risk_factors <= 1
    confidence = max(0.6, 1.0 - (risk_factors * 0.1))
    
    insights = generate_water_quality_insights(data)
    
    return jsonify({
        "prediction": 1 if is_potable else 0,
        "status": "POTABLE" if is_potable else "NOT POTABLE",
        "confidence": float(confidence),
        "probabilities": {
            "potable": float(confidence if is_potable else 1 - confidence),
            "not_potable": float(1 - confidence if is_potable else confidence)
        },
        "top_influencing_factors": ["Demo mode - using rule-based analysis"],
        "quality_insights": insights,
        "input_parameters": data,
        "model_used": "Demo Rule-Based System",
        "warning": "Running in demo mode - ML model not loaded"
    })

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
    
    # Trihalomethanes analysis
    if data['Trihalomethanes'] > 80:
        insights.append("🚫 High Trihalomethanes - potential health risk")
    elif data['Trihalomethanes'] > 0:
        insights.append("✅ Trihalomethanes within acceptable limits")
    
    return insights

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch predictions"""
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({"error": "No samples provided"}), 400
        
        # Demo mode if model not loaded
        if not model_loaded:
            results = []
            for sample in samples:
                demo_result = demo_prediction(sample).get_json()
                results.append({
                    "prediction": demo_result["prediction"],
                    "confidence": demo_result["confidence"],
                    "status": demo_result["status"]
                })
            
            return jsonify({
                "batch_id": f"demo_batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                "results": results,
                "total_samples": len(results),
                "mode": "demo"
            })
        
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
            "total_samples": len(results),
            "mode": "ml_model"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint to check available features"""
    return jsonify({
        "feature_names": feature_names,
        "total_features": len(feature_names),
        "model_loaded": model_loaded
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask server on port {port}...")
    print(f"🌐 Access the API at: http://127.0.0.1:{port}")
    print(f"🔍 Health check: http://127.0.0.1:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=True)