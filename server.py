# from fastapi import FastAPI
# import pickle

# with open("XGB_attack_all.pkl", "rb") as f:
#     model = pickle.load(f)

# class_names = ['Generic', 'Shellcode', 'Exploits', 'Reconnaissance', 'Backdoor', 
#                'Normal', 'Analysis', 'Fuzzers', 'DoS', 'Worms']

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "ML model deployment"}

# @app.post("/predict")
# def predict(data: dict):
#     """
#     Predict the class of the input data using the loaded model.
#     """
#     # Preprocess the input data
#     data = [data['data']]
    
#     # Make prediction
#     prediction = model.predict(data)
    
#     # Map prediction to class name
#     class_name = class_names[prediction[0]]
    
#     return {"prediction": class_name}

# ------------------------------------------------------------------------------
# from fastapi import FastAPI
# from pydantic import BaseModel, Field # Use Pydantic for input validation
# import torch
# import torch.nn as nn
# import numpy as np
# import os
# from typing import List

# # --- Define the Model Architecture (Must match train4.py) ---
# class MultiTaskNeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, hidden_size_2, num_classes_attack, num_classes_risk):
#         super(MultiTaskNeuralNet, self).__init__()
#         self.shared_l1 = nn.Linear(input_size, hidden_size)
#         self.shared_l2 = nn.Linear(hidden_size, hidden_size_2)
#         self.relu = nn.ReLU()
#         self.attack_out = nn.Linear(hidden_size_2, num_classes_attack)
#         self.risk_out = nn.Linear(hidden_size_2, num_classes_risk)

#     def forward(self, x):
#         x = self.relu(self.shared_l1(x))
#         x = self.relu(self.shared_l2(x))
#         return self.attack_out(x), self.risk_out(x)

# # --- Configuration ---
# MODEL_PATH = "multitask_nn_selected_features.pth" # Path to your PyTorch model
# INPUT_SIZE = 10  # Input size for the RFE model (10 features)
# HIDDEN_SIZE = 128
# HIDDEN_SIZE_2 = 64
# # IMPORTANT: Verify this list matches the LabelEncoder order from train4.py
# # You might need to load the LabelEncoder or hardcode the correct order.
# # Example order (replace if different):
# ATTACK_CLASS_NAMES = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 
#                       'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms']
# NUM_ATTACK_CLASSES = len(ATTACK_CLASS_NAMES)
# # Based on severity_map in train4.py: max value is 4, so classes are 0, 1, 2, 3, 4
# NUM_RISK_CLASSES = 5
# # Optional: Map risk indices to meaningful names
# RISK_LEVEL_NAMES = {0: "Normal", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}

# # --- Load PyTorch Model ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MultiTaskNeuralNet(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE_2, NUM_ATTACK_CLASSES, NUM_RISK_CLASSES)

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# try:
#     # Load the state dictionary, ensuring it's mapped to the correct device
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#     model.to(device) # Move the model to the device
#     model.eval() # Set model to evaluation mode (important!)
#     print(f"PyTorch model loaded successfully from {MODEL_PATH} onto {device}.")
# except Exception as e:
#     raise RuntimeError(f"Error loading PyTorch model: {e}")

# # --- FastAPI App ---
# app = FastAPI()

# # --- Input Data Model ---
# class InputData(BaseModel):
#     # Define the expected input structure and add an example
#     data: List[float] = Field(..., example=[0.1] * INPUT_SIZE, description=f"List of {INPUT_SIZE} feature values")

# @app.get("/")
# def read_root():
#     # Update the root message
#     return {"message": "Multi-Task Intrusion Detection API"}

# @app.post("/predict")
# def predict(item: InputData): # Use the Pydantic model for input validation
#     """
#     Predict the attack type and risk level of the input data
#     using the multitask_nn_selected_features model.
#     """
#     try:
#         # 1. Get data from the validated Pydantic model
#         input_features = item.data

#         # 2. Validate input length (Pydantic helps, but explicit check is good)
#         if len(input_features) != INPUT_SIZE:
#             # This error might not be reached if Pydantic validation is strict enough, but good practice
#             return {"error": f"Invalid input length. Expected {INPUT_SIZE} features, got {len(input_features)}."}

#         # 3. Convert to PyTorch Tensor
#         # Add batch dimension [input_features], set dtype, move to device
#         input_tensor = torch.tensor([input_features], dtype=torch.float32).to(device)

#         # 4. Make prediction using the PyTorch model
#         with torch.no_grad(): # Disable gradient calculation for inference
#             out_attack, out_risk = model(input_tensor)

#         # 5. Get predicted indices by finding the max logit/probability index
#         pred_idx_attack = torch.argmax(out_attack, dim=1).item()
#         pred_idx_risk = torch.argmax(out_risk, dim=1).item()

#         # 6. Map indices to names/levels
#         predicted_attack_name = ATTACK_CLASS_NAMES[pred_idx_attack]
#         # Use .get() for safer dictionary access in case of unexpected index
#         predicted_risk_level_name = RISK_LEVEL_NAMES.get(pred_idx_risk, "Unknown Risk Index")

#         # 7. Return both predictions
#         return {
#             "predicted_attack_type": predicted_attack_name,
#             "predicted_risk_level_index": pred_idx_risk,
#             "predicted_risk_level_name": predicted_risk_level_name
#         }

#     except Exception as e:
#         # Log the error for server-side debugging
#         print(f"Error during prediction: {e}")
#         import traceback
#         traceback.print_exc()
#         # Return a generic error message to the client
#         return {"error": f"An error occurred during prediction."}

# To run: uvicorn server:app --reload
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import pickle
# import numpy as np
# import pandas as pd
# import os
# import sys
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# from typing import List

# # --- Configuration & Model Loading ---
# MODEL_DIR = "models/"
# STAGE1_SCALER_PATH = os.path.join(MODEL_DIR, "stage1_ocsvm_scaler.p")
# STAGE1_MODEL_PATH = os.path.join(MODEL_DIR, "stage1_ocsvm_100k.p")
# STAGE2_SCALER_PATH = os.path.join(MODEL_DIR, "stage2_rf_scaler.p")
# STAGE2_MODEL_PATH = os.path.join(MODEL_DIR, "stage2_rf.p")

# # Thresholds (from train_comb.py, "Full model performance" section for rf_model_extra_feature)
# TAU_B = -0.10866126632226556
# TAU_M = 0.60
# TAU_U = -0.00015517209205490046

# try:
#     with open(STAGE1_SCALER_PATH, "rb") as f:
#         stage1_scaler = pickle.load(f)
#     with open(STAGE1_MODEL_PATH, "rb") as f:
#         stage1_model = pickle.load(f)
#     with open(STAGE2_SCALER_PATH, "rb") as f:
#         stage2_scaler = pickle.load(f)
#     with open(STAGE2_MODEL_PATH, "rb") as f:
#         stage2_model = pickle.load(f)
# except FileNotFoundError as e:
#     print(f"Error: Model file not found. {e}")
#     raise RuntimeError(f"Could not load model files: {e}")
# except Exception as e:
#     print(f"Error loading models: {e}")
#     raise RuntimeError(f"Error loading models: {e}")

# N_EXPECTED_FEATURES = stage1_scaler.n_features_in_
# print(f"Models loaded. Expecting {N_EXPECTED_FEATURES} input features.")

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*", "null"], # Thêm "null" nếu cần cho file:///
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class InputData(BaseModel):
#     features: List[float] = Field(
#         ...,
#         example=[0.1] * N_EXPECTED_FEATURES, # Example will adjust to actual feature count
#         description=f"List of {N_EXPECTED_FEATURES} feature values."
#     )

# @app.get("/")
# def read_root():
#     return {"message": "Multi-Stage Intrusion Detection Pipeline API"}

# @app.post("/predict")
# def predict(item: InputData):
#     try:
#         input_features_list = item.features

#         if len(input_features_list) != N_EXPECTED_FEATURES:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid input length. Expected {N_EXPECTED_FEATURES} features, got {len(input_features_list)}."
#             )

#         # Convert to NumPy array and reshape for single sample
#         x_input_np = np.array(input_features_list).reshape(1, -1)

#         # --- Stage 1: Anomaly Detection (OCSVM) ---
#         x_scaled_for_stage1 = stage1_scaler.transform(x_input_np)
#         proba_1_score = -stage1_model.decision_function(x_scaled_for_stage1) # Single score in an array
        
#         # Initial prediction: Benign or Attack
#         # Ensure pred_1 is an array of strings, even for a single sample
#         pred_1 = np.array(["Attack"] * x_input_np.shape[0], dtype=object) 
#         pred_1[proba_1_score < TAU_B] = "Benign"

#         # Initialize final prediction array
#         y_final_pred = pred_1.copy()

#         # --- Stage 2: Multi-class Classification (RandomForest) ---
#         # Process only samples predicted as "Attack" by Stage 1
#         attack_mask_stage1 = (pred_1 == "Attack")
        
#         # pred_2 will store Stage 2 predictions for "Attack" samples
#         # Initialize with a placeholder or handle empty case
#         pred_2_for_attacks = np.array([], dtype=object)


#         if np.any(attack_mask_stage1):
#             x_attack_original_np = x_input_np[attack_mask_stage1]
#             proba_1_for_attack_samples = proba_1_score[attack_mask_stage1]

#             x_attack_scaled_for_stage2 = stage2_scaler.transform(x_attack_original_np)
            
#             # Combine scaled features with Stage 1 scores for Stage 2 model input
#             input_for_stage2 = np.column_stack((
#                 x_attack_scaled_for_stage2,
#                 proba_1_for_attack_samples.reshape(-1, 1) # Ensure proba_1 is 2D for hstack
#             ))
            
#             proba_2_raw = stage2_model.predict_proba(input_for_stage2)
            
#             max_proba_stage2 = np.max(proba_2_raw, axis=1)
#             argmax_proba_stage2 = np.argmax(proba_2_raw, axis=1)

#             # Initialize pred_2_for_attacks with "Unknown"
#             pred_2_for_attacks = np.array(["Unknown"] * proba_2_raw.shape[0], dtype=object)
#             # Apply TAU_M threshold
#             confident_mask_stage2 = max_proba_stage2 > TAU_M
#             pred_2_for_attacks[confident_mask_stage2] = stage2_model.classes_[argmax_proba_stage2[confident_mask_stage2]]
            
#             # Update y_final_pred for "Attack" samples with Stage 2 results
#             y_final_pred[attack_mask_stage1] = pred_2_for_attacks

#         # --- Extension Stage: Zero-Day Detection ---
#         # Process samples that are "Unknown" after Stage 2
#         # (These were originally "Attack" from Stage 1, then "Unknown" from Stage 2)
#         unknown_after_stage2_mask = (y_final_pred == "Unknown")
        
#         if np.any(unknown_after_stage2_mask):
#             # Scores for these "Unknown" samples are their original Stage 1 scores
#             proba_3_input_scores = proba_1_score[unknown_after_stage2_mask]
            
#             pred_3_for_unknowns = np.array(["Unknown"] * proba_3_input_scores.shape[0], dtype=object)
#             pred_3_for_unknowns[proba_3_input_scores < TAU_U] = "Benign"
            
#             # Update y_final_pred for these "Unknown" samples with Extension Stage results
#             y_final_pred[unknown_after_stage2_mask] = pred_3_for_unknowns

#         # The final prediction is a single string for the single input sample
#         final_prediction_label = y_final_pred[0]

#         return {"prediction": final_prediction_label}

#     except HTTPException as http_exc:
#         raise http_exc # Re-raise HTTPException to let FastAPI handle it
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         import traceback
#         traceback.print_exc()
#         # Raise HTTPException for other errors
#         raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

import asyncio
import csv
import io
import os
import pickle
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Import CORS
from pydantic import BaseModel, Field # Import Field for InputData
import numpy as np # Import numpy


# --- Configuration & Model Paths (Adapt from your existing setup) ---
MODEL_DIR = "models/" # Ensure this path is correct relative to where server.py runs
STAGE1_SCALER_PATH = os.path.join(MODEL_DIR, "stage1_ocsvm_scaler.p")
STAGE1_MODEL_PATH = os.path.join(MODEL_DIR, "stage1_ocsvm_100k.p")
STAGE2_SCALER_PATH = os.path.join(MODEL_DIR, "stage2_rf_scaler.p")
STAGE2_MODEL_PATH = os.path.join(MODEL_DIR, "stage2_rf.p")

# Expected number of features after parsing a row
N_EXPECTED_FEATURES = 67 # **** IMPORTANT: Set this to your model's expected input feature count ****

TAU_B = -0.10866126632226556
TAU_M = 0.60
TAU_U = -0.00015517209205490046

app = FastAPI()

# --- Global In-Memory Store for Monitoring ---
# For a production app, consider a more robust store (e.g., Redis, database, or per-session objects)
monitoring_data_store: Dict[str, Any] = {
    "csv_rows": [],          # List of rows (each row is a list of strings)
    "predictions": [],       # List of prediction objects
    "current_index": 0,      # Current line index to process
    "is_active": False,      # Is monitoring currently active?
    "file_name": None,
    "total_lines": 0
}

# --- Model Loading ---
# These will hold your loaded models/scalers
stage1_scaler_g = None
stage1_model_g = None
stage2_scaler_g = None
stage2_model_g = None

def load_all_models():
    global stage1_scaler_g, stage1_model_g, stage2_scaler_g, stage2_model_g
    try:
        with open(STAGE1_SCALER_PATH, "rb") as f:
            stage1_scaler_g = pickle.load(f) # Uncomment and complete
        with open(STAGE1_MODEL_PATH, "rb") as f:
            stage1_model_g = pickle.load(f) # Uncomment and complete
        with open(STAGE2_SCALER_PATH, "rb") as f:
            stage2_scaler_g = pickle.load(f) # Uncomment and complete
        with open(STAGE2_MODEL_PATH, "rb") as f:
            stage2_model_g = pickle.load(f)
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # Handle critical error - perhaps exit or prevent app from fully starting
        raise RuntimeError(f"Could not load models: {e}")

@app.on_event("startup")
async def startup_event():
    load_all_models()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"], # Allows all origins, and "null" for file:///
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Core Prediction Logic (Revised to match multi-stage logic) ---
def run_prediction_pipeline(features: List[float]) -> str:
    """
    Processes a single feature vector through the multi-stage pipeline.
    """
    if not all([stage1_scaler_g, stage1_model_g, stage2_scaler_g, stage2_model_g]):
        # This check might be redundant if startup_event handles load failure by exiting
        print("Error: Models not loaded properly.")
        raise RuntimeError("Models not loaded. Cannot perform prediction.")

    x_input_np = np.array([features]) # Model expects 2D array

    try:
        # --- Stage 1: Anomaly Detection (OCSVM) ---
        x_scaled_for_stage1 = stage1_scaler_g.transform(x_input_np)
        # decision_function returns scores; your train.py uses negative, so higher is more anomalous
        proba_1_score_array = -stage1_model_g.decision_function(x_scaled_for_stage1)
        proba_1_score_single = proba_1_score_array[0] # Get the single score

        pred_1 = np.array(["Attack"] * x_input_np.shape[0], dtype=object)
        if proba_1_score_single < TAU_B:
            pred_1[0] = "Benign"

        y_final_pred = pred_1.copy()

        # --- Stage 2: Multi-class Classification (RandomForest) ---
        attack_mask_stage1 = (pred_1[0] == "Attack")

        if attack_mask_stage1:
            # x_attack_original_np = x_input_np # Since it's a single sample already filtered
            proba_1_for_attack_samples = proba_1_score_array # Use the array form for column_stack

            x_attack_scaled_for_stage2 = stage2_scaler_g.transform(x_input_np)

            input_for_stage2 = np.column_stack((
                x_attack_scaled_for_stage2,
                proba_1_for_attack_samples.reshape(-1, 1)
            ))

            proba_2_raw_all_classes = stage2_model_g.predict_proba(input_for_stage2)[0] # Probabilities for the single sample
            
            max_proba_stage2 = np.max(proba_2_raw_all_classes)
            argmax_idx_stage2 = np.argmax(proba_2_raw_all_classes)

            pred_2_single_attack = "Unknown" # Default for this stage
            if max_proba_stage2 > TAU_M:
                pred_2_single_attack = stage2_model_g.classes_[argmax_idx_stage2]
            
            y_final_pred[0] = pred_2_single_attack # Update the final prediction

        # --- Extension Stage: Zero-Day Detection ---
        unknown_after_stage2_mask = (y_final_pred[0] == "Unknown")

        if unknown_after_stage2_mask:
            # proba_3_input_scores = proba_1_score_array # Use the array form
            pred_3_single_unknown = "Unknown" # Default for this stage
            if proba_1_score_single < TAU_U: # Use the single score for comparison
                pred_3_single_unknown = "Benign"
            
            y_final_pred[0] = pred_3_single_unknown

        final_prediction_label = y_final_pred[0]
        return final_prediction_label

    except Exception as e:
        print(f"Error during prediction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: Prediction failed ({str(e)})"


# --- API Endpoints for Monitoring ---
@app.post("/upload_monitoring_csv")
async def upload_csv_for_monitoring(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    contents = await file.read()
    
    # Reset monitoring state
    monitoring_data_store["csv_rows"] = []
    monitoring_data_store["predictions"] = []
    monitoring_data_store["current_index"] = 0
    monitoring_data_store["is_active"] = False
    monitoring_data_store["file_name"] = file.filename
    monitoring_data_store["total_lines"] = 0

    try:
        buffer = io.StringIO(contents.decode())
        reader = csv.reader(buffer)
        header = next(reader, None) # Skip header if present

        for row_strings in reader:
            if any(field.strip() for field in row_strings): # Ensure row is not entirely empty
                monitoring_data_store["csv_rows"].append(row_strings)
        
        monitoring_data_store["total_lines"] = len(monitoring_data_store["csv_rows"])
        if monitoring_data_store["total_lines"] == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty or contains no valid data rows.")

        return {
            "message": f"CSV '{file.filename}' uploaded. Contains {monitoring_data_store['total_lines']} data rows. Ready for monitoring.",
            "fileName": file.filename,
            "rowCount": monitoring_data_store["total_lines"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@app.post("/start_monitoring")
async def start_monitoring_session():
    if not monitoring_data_store["csv_rows"]:
        raise HTTPException(status_code=400, detail="No CSV file uploaded for monitoring.")
    
    monitoring_data_store["predictions"] = [] # Clear previous run's predictions
    monitoring_data_store["current_index"] = 0
    monitoring_data_store["is_active"] = True
    print(f"Monitoring started for {monitoring_data_store['file_name']}. Total lines: {monitoring_data_store['total_lines']}")
    return {"message": "System monitoring started."}

@app.post("/stop_monitoring")
async def stop_monitoring_session():
    monitoring_data_store["is_active"] = False
    print("Monitoring stopped by request.")
    return {"message": "System monitoring stopped."}

@app.get("/get_monitoring_update")
async def get_monitoring_update():
    if not monitoring_data_store["is_active"]:
        return {
            "status": "idle",
            "all_predictions": monitoring_data_store["predictions"],
            "latest_prediction": None,
            "processed_lines": monitoring_data_store["current_index"],
            "total_lines": monitoring_data_store["total_lines"]
        }

    if monitoring_data_store["current_index"] >= monitoring_data_store["total_lines"]:
        monitoring_data_store["is_active"] = False # Auto-stop
        return {
            "status": "finished",
            "all_predictions": monitoring_data_store["predictions"],
            "latest_prediction": None,
            "message": "All lines processed.",
            "processed_lines": monitoring_data_store["current_index"],
            "total_lines": monitoring_data_store["total_lines"]
        }

    current_row_str_list = monitoring_data_store["csv_rows"][monitoring_data_store["current_index"]]
    latest_prediction_obj = {
        "line_index": monitoring_data_store["current_index"],
        "input_row_str": ", ".join(current_row_str_list), # For display
        "prediction": "Error: Processing failed",
        "is_error": True,
        "padded_count": 0, # Initialize padded_count
        "truncated_count": 0 # Initialize truncated_count
    }

    try:
        features_float_original = [float(val_str.strip()) for val_str in current_row_str_list]
        features_float = list(features_float_original) # Work with a copy

        padded_count = 0
        truncated_count = 0
        
        original_len = len(features_float)

        if original_len < N_EXPECTED_FEATURES:
            padded_count = N_EXPECTED_FEATURES - original_len
            features_float.extend([0.0] * padded_count)
        elif original_len > N_EXPECTED_FEATURES:
            truncated_count = original_len - N_EXPECTED_FEATURES
            features_float = features_float[:N_EXPECTED_FEATURES]

        latest_prediction_obj["padded_count"] = padded_count
        latest_prediction_obj["truncated_count"] = truncated_count
        
        prediction_label = run_prediction_pipeline(features_float)
        latest_prediction_obj["prediction"] = prediction_label
        latest_prediction_obj["is_error"] = prediction_label.startswith("Error:")

    except ValueError as ve:
        latest_prediction_obj["prediction"] = f"Error: Invalid data in row (ValueError: {ve})."
    except Exception as e:
        latest_prediction_obj["prediction"] = f"Error: Prediction failed ({e})."
    
    monitoring_data_store["predictions"].append(latest_prediction_obj)
    monitoring_data_store["current_index"] += 1

    return {
        "status": "processing",
        "all_predictions": monitoring_data_store["predictions"], # Send the whole history
        "latest_prediction": latest_prediction_obj,
        "processed_lines": monitoring_data_store["current_index"],
        "total_lines": monitoring_data_store["total_lines"]
    }

# --- Your existing /predict endpoint for single/batch manual input (can remain) ---
class InputData(BaseModel):
    features: List[float] = Field(..., example=[0.1] * N_EXPECTED_FEATURES, description=f"List of {N_EXPECTED_FEATURES} feature values.") # Add Field for example

@app.post("/predict")
async def predict_manual_input(item: InputData):
    if len(item.features) != N_EXPECTED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input. Expected {N_EXPECTED_FEATURES} features, got {len(item.features)}"
        )
    try:
        prediction = run_prediction_pipeline(item.features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add other existing endpoints if any
