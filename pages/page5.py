# import pandas as pd
# import numpy as np
# import streamlit as st
# import joblib

# # Load trained model and feature means
# best_xg_model = joblib.load("best_xg_model.joblib")  
# feature_means = pd.read_csv("data/train_data_stats.csv", index_col=0).squeeze()  
# normal_means = pd.read_csv("data/normal_traffic_means.csv", index_col=0).squeeze()  

# # Define class labels
# class_labels = {0: "Analysis", 1: "Backdoor", 2: "DoS", 3: "Exploits", 4: "Fuzzers", 
#                 5: "Generic", 6: "Normal", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"} 

# def get_prediction_with_confidence(input_data):
#     # Overall means prediction
#     input1 = pd.DataFrame([feature_means], index=[0])
#     input1[top_features] = [input_data]
#     pred1 = best_xg_model.predict_proba(input1)[0]
    
#     # Normal traffic means prediction
#     input2 = pd.DataFrame([normal_means], index=[0])
#     input2[top_features] = [input_data]
#     pred2 = best_xg_model.predict_proba(input2)[0]
    
#     # Set thresholds based on actual precision scores from test results
#     class_thresholds = {
#         0: 0.60,  # Analysis (precision: 0.60)
#         1: 0.58,  # Backdoor (precision: 0.58)
#         2: 0.45,  # DoS (precision: 0.45)
#         3: 0.80,  # Exploits (precision: 0.80)
#         4: 0.90,  # Fuzzers (precision: 0.90)
#         5: 0.95,  # Generic (precision: 1.00)
#         6: 0.95,  # Normal (precision: 0.97)
#         7: 0.90,  # Reconnaissance (precision: 0.97)
#         8: 0.95,  # Shellcode (precision: 0.97)
#         9: 0.95   # Worms (precision: 0.99)
#     }

#     # Weighted combination of predictions
#     final_probs = (pred1 * 0.8 + pred2 * 0.2)
    
#     # Get initial prediction
#     prediction = np.argmax(final_probs)
#     confidence = final_probs[prediction]
    
#     # Attack severity weights based on class characteristics
#     attack_severity = {
#         "Normal": 0,      # Class 6
#         "Generic": 1,     # Class 5
#         "Fuzzers": 2,     # Class 4
#         "Analysis": 3,    # Class 0
#         "Reconnaissance": 4, # Class 7
#         "DoS": 5,        # Class 2
#         "Backdoor": 6,    # Class 1
#         "Exploits": 7,    # Class 3
#         "Shellcode": 8,   # Class 8
#         "Worms": 9       # Class 9
#     }

#     # Confidence threshold adjustment based on recall scores
#     recall_adjustment = {
#         0: 0.59,  # Analysis recall
#         1: 0.66,  # Backdoor recall
#         2: 0.58,  # DoS recall
#         3: 0.63,  # Exploits recall
#         4: 0.90,  # Fuzzers recall
#         5: 0.99,  # Generic recall
#         6: 0.92,  # Normal recall
#         7: 0.85,  # Reconnaissance recall
#         8: 1.00,  # Shellcode recall
#         9: 1.00   # Worms recall
#     }

#     # If confidence is below threshold, look for alternative predictions
#     if confidence < class_thresholds[prediction] * recall_adjustment[prediction]:
#         sorted_indices = np.argsort(final_probs)[::-1]
#         for idx in sorted_indices:
#             adjusted_threshold = class_thresholds[idx] * recall_adjustment[idx]
#             if final_probs[idx] >= adjusted_threshold:
#                 prediction = idx
#                 confidence = final_probs[idx]
#                 break
    
#     # Calculate risk score using both probability and severity
#     risk_score = sum(prob * attack_severity[class_labels[i]] for i, prob in enumerate(final_probs))
    
#     # Normalize risk score to 0-10 scale
#     risk_score = min(10, risk_score / len(attack_severity) * 10)
    
#     return prediction, confidence, final_probs, risk_score


# # Main UI
# st.markdown("# Intrusion Detection")

# # Input fields for top 8 selected features
# top_features = ['sttl', 'service', 'ct_dst_sport_ltm', 'is_sm_ips_ports', 
#                 'swin', 'ct_flw_http_mthd', 'ct_state_ttl', 'sloss']
# input_data = []

# feature_descriptions = {
#     'sttl': 'Source to destination time to live',
#     'service': 'Network service type',
#     'ct_dst_sport_ltm': 'Count of connections with same dst address and src port in last 100 connections',
#     'is_sm_ips_ports': 'If source equals to destination IP addresses and port numbers',
#     'swin': 'Source TCP window size',
#     'ct_flw_http_mthd': 'Count of flows that has methods such as Get and Post in http service',
#     'ct_state_ttl': 'Count of connections with same state and TTL',
#     'sloss': 'Source packets retransmitted or dropped'
# }

# for feature in top_features:
#     value = st.number_input(f"Enter {feature}:", 
#                           min_value=0.0, 
#                           value=feature_means[feature],
#                           help=feature_descriptions.get(feature, ""))

#     input_data.append(value)

# if st.button("Predict"):
#     pred, conf, probs, risk_score = get_prediction_with_confidence(input_data)
    
#     # Display prediction with confidence
#     st.write(f"Prediction: {class_labels[pred]}")
#     st.write(f"Confidence: {conf:.2%}")
    
#     # Display risk level
#     risk_level = "Low" if risk_score < 3 else "Medium" if risk_score < 6 else "High"
#     risk_color = {"Low": "green", "Medium": "orange", "High": "red"}
#     st.markdown(f"Risk Level: <span style='color: {risk_color[risk_level]}'>{risk_level}</span>", 
#                unsafe_allow_html=True)
    
#     # Adjust confidence thresholds based on class
#     class_confidence_threshold = {
#         "Normal": 0.97,
#         "DoS": 0.45,
#         "Analysis": 0.60,
#         "Backdoor": 0.58,
#         "Exploits": 0.80,
#         "Fuzzers": 0.90,
#         "Generic": 0.95,
#         "Reconnaissance": 0.97,
#         "Shellcode": 0.97,
#         "Worms": 0.99
#     }
    
#     predicted_class = class_labels[pred]
#     if conf < class_confidence_threshold[predicted_class]:
#         st.warning(f"Low confidence prediction for {predicted_class}. Consider additional verification.")
    
#     # Show top 3 possibilities with probability bars
#     st.write("Top 3 possible classifications:")
#     top3_indices = np.argsort(probs)[-3:][::-1]
#     for idx in top3_indices:
#         prob_percentage = probs[idx] * 100
#         st.progress(prob_percentage / 100)
#         st.write(f"{class_labels[idx]}: {prob_percentage:.1f}%")
    
#     # Additional context for security analysts
#     if class_labels[pred] != "Normal":
#         st.warning("""
#         Recommended Actions:
#         1. Verify the input features
#         2. Check related network logs
#         3. Consider blocking suspicious IPs
#         """)
        
#     # Feature importance for this prediction
#     st.write("Most influential features for this prediction:")
#     feature_values = pd.DataFrame({
#         'Feature': top_features,
#         'Value': input_data
#     })
#     st.table(feature_values)

#     # Add timestamp for logging
#     st.write(f"Prediction made at: {pd.Timestamp.now()}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load feature statistics
feature_means = pd.read_csv("data/train_data_stats.csv", index_col=0, header=0)

# Update the feature list
all_features = feature_means.index.tolist()


class_labels = {0: "Analysis", 1: "Backdoor", 2: "DoS", 3: "Exploits", 4: "Fuzzers", 
                 5: "Generic", 6: "Normal", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"} 

import numpy as np

# Dummy model for demonstration (Replace with your actual model)
def get_prediction_with_confidence(input_data):
    """
    Takes user input features, runs a prediction using a trained model,
    and returns the predicted class, confidence, probabilities, and risk score.
    """
    # Convert input to NumPy array
    input_array = np.array(input_data).reshape(1, -1)

    # Load your trained model
    model = joblib.load("best_xg_model.joblib")

    # Make actual prediction using the trained model
    predicted_class = model.predict(input_array)[0]  # Get class prediction

    # Get class probabilities (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_array)[0]  # Get probability distribution
        confidence = np.max(probabilities)  # Highest probability as confidence
    else:
        probabilities = None
        confidence = 1.0  # Assume 100% confidence if no probabilities available

    # Risk score (Define logic)
    risk_score = confidence * 100  # Example formula

    return predicted_class, confidence, probabilities, risk_score


# Streamlit App Title
st.title("🛡️ Intrusion Detection System - Threat Analysis")

# Sidebar for User Input
st.sidebar.header("🔹 Enter Feature Values")
st.sidebar.write("Adjust feature values or keep the defaults (mean values).")

# Collect user inputs with default mean values
input_data = []
for feature in all_features:
    value = st.sidebar.number_input(
        f"{feature}:", 
        min_value=0.0, 
        value = feature_means.loc[feature, '0'],  # Explicit column name
        help=f"Mean: {feature_means.loc[feature, '0']:.4f}"
    )
    input_data.append(value)

# Submit Button
if st.sidebar.button("Detect 🚀"):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data], columns=all_features)

    # Call prediction function
    pred, conf, probs, risk_score = get_prediction_with_confidence(input_data)

    # Display results
    st.subheader("📊 Detection Results")
    st.markdown(f"**🟢 Prediction:** `{class_labels[pred]}`")
    st.markdown(f"**🔹 Confidence:** `{conf:.2%}`")

    # Confidence Progress Bar
    st.progress(int(conf * 100))

    # Show Risk Score (if applicable)
    if risk_score:
        st.markdown(f"**⚠️ Threat Risk Score:** `{risk_score:.2f}`")

    # Show Probability Distribution (if applicable)
    if probs is not None:
        st.subheader("📌 Probability Distribution")
        prob_df = pd.DataFrame(probs, index=list(class_labels.values()), columns=["Probability"])
        st.bar_chart(prob_df)
