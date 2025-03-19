import flwr as fl
import os
import numpy as np
import time
import plot
import logging
import json
from client import DigitalTwin
from typing import List, Tuple, Optional, Dict
from flwr.common import parameters_to_ndarrays
from loader import ModelLoader 
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import GrpcBridgeClosed
# At the start of your script
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GrpcFilter(logging.Filter):
    def filter(self, record):
        # Filter out both GrpcBridgeClosed messages and the exception traceback
        if "GrpcBridgeClosed" in record.getMessage():
            return False
        if "Exception iterating responses" in record.getMessage():
            return False
        return True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add filter to both the root logger and the grpc logger specifically
root_logger = logging.getLogger()
root_logger.addFilter(GrpcFilter())

# Also add the filter to the grpc logger specifically
grpc_logger = logging.getLogger('grpc')
grpc_logger.addFilter(GrpcFilter())

# Set grpc logger to a higher level
logging.getLogger('grpc').setLevel(logging.ERROR)


# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ✅ Define global Digital Twin variable
digital_twin = None

def start_server_with_retry(max_retries=3, retry_delay=5):
    """Start the Flower server with retry mechanism."""
    retries = 0
    while retries < max_retries:
        try:
            history = fl.server.start_server(
                server_address="0.0.0.0:8601",
                strategy=get_server_strategy(),
                config=fl.server.ServerConfig(num_rounds=5)
            )
            print("✅ Server completed successfully!")
            return history
            
        except GrpcBridgeClosed:
            print(f"⚠️ gRPC bridge closed, attempt {retries + 1} of {max_retries}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            
        except Exception as e:
            print(f"❌ Unexpected server error: {e}")
            raise
            
    print("❌ Max retries reached, server failed to start")
    return None

def validate_json_file(filepath):
    """Validate JSON file before loading."""
    try:
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            return False
            
        with open(filepath, "r") as f:
            content = f.read()
            print(f"File size: {len(content)} bytes")
            
            try:
                # Try parsing the content
                json.loads(content)
                return True
            except json.JSONDecodeError as je:
                print(f"❌ JSON validation error at position {je.pos}: {je.msg}")
                # Show the problematic section
                start = max(0, je.pos - 50)
                end = min(len(content), je.pos + 50)
                print(f"Problematic section: ...{content[start:end]}...")
                return False
                
    except Exception as e:
        print(f"❌ Error validating file: {str(e)}")
        return False

def load_digital_twin():
    global digital_twin

    input_shape = (42,)  # Ensure this matches the client
    model = ModelLoader.get_model(input_shape)

    # Load model weights
    if os.path.exists("saved_model.h5"):
        model.load_weights("saved_model.h5")
        print("✅ Loaded saved model weights.")

    # Create Digital Twin
    digital_twin = DigitalTwin(client_id="server")
    digital_twin.model = model  # Attach model to Digital Twin

    # Validate and load normal behavior
    json_file = "saved_model_fixed.json"
    if validate_json_file(json_file):
        try:
            with open(json_file, "r") as f:
                normal_behavior_list = json.load(f)
            
            # Convert to NumPy array
            digital_twin.normal_behavior = np.array(normal_behavior_list)    
            print(f"✅ Digital Twin normal behavior loaded with shape: {digital_twin.normal_behavior.shape}")
        except Exception as e:
            print(f"❌ Error loading normal behavior: {e}")
            # Initialize with default values
            digital_twin.normal_behavior = np.zeros((1, 42))
            print("⚠️ Initialized with default values")
    else:
        print("🚨 Warning: JSON file validation failed!")
        # Initialize with default values
        digital_twin.normal_behavior = np.zeros((1, 42))
        print("⚠️ Initialized with default values")

def weighted_average(metrics):
    total_examples = 0
    federated_metrics = {k: 0 for k in metrics[0][1].keys()}
    for num_examples, m in metrics:
        if num_examples == 0:  # Skip this client if no examples
            continue
        for k, v in m.items():
            federated_metrics[k] += num_examples * v
        total_examples += num_examples
    
    if total_examples == 0:
        # Handle case where there are no valid examples
        return {k: 0 for k in federated_metrics}  # Default to 0 or skip aggregation
    
    return {k: v / total_examples for k, v in federated_metrics.items()}

def validate_updates_with_digital_twin(updates):
    """
    Validate client updates using the Digital Twin anomaly detection model.
    If updates are valid, the Digital Twin is updated dynamically.
    """
    
    # Load the Digital Twin model before starting the server
    load_digital_twin()
    global digital_twin

    validated_updates = []
    valid_data_points = []  # Store valid mean & std values for updating the Digital Twin

    for client_id, update in enumerate(updates):
        try:
            # Deserialize protobuf tensors to NumPy arrays
            update_ndarrays = parameters_to_ndarrays(update)

            # Extract statistical features from model weights
            mean_weights = np.mean([np.mean(w) for w in update_ndarrays])
            std_weights = np.std([np.std(w) for w in update_ndarrays])
            feature_vector = np.array([mean_weights, std_weights])

            # Validate the update using the Digital Twin model
            is_valid = digital_twin.validate_data(feature_vector)

            if is_valid:
                validated_updates.append(update)
                valid_data_points.append(feature_vector)
                print(f"[Validation ✅] Client {client_id}: Update accepted (Mean={mean_weights:.2f}, Std={std_weights:.2f})")
            else:
                print(f"[Validation ❌] Client {client_id}: Update rejected (Mean={mean_weights:.2f}, Std={std_weights:.2f})")

        except Exception as e:
            print(f"[ERROR] Failed to process update from client {client_id}: {e}")
            continue

    # Update Digital Twin if we have new valid updates
    if valid_data_points:
        try:
            new_normal_behavior = np.array(valid_data_points)
            digital_twin.update_normal_behavior(new_normal_behavior)
            print(f"[Digital Twin 🔄] Updated with {len(valid_data_points)} new valid updates.")
        except Exception as e:
            print(f"[ERROR] Failed to update Digital Twin: {e}")

    print(f"[Validation Summary] Accepted: {len(validated_updates)}, Rejected: {len(updates) - len(validated_updates)}")
    return validated_updates

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        # Initialize Digital Twin with the same parameters as client
        self.digital_twin = DigitalTwin(client_id="server_twin", threshold_factor=2.0)
        self.initialize_digital_twin()

    def initialize_digital_twin(self):
        """Initialize the Digital Twin with proper configuration."""
        try:
            json_file_path = "saved_model_fixed.json"
            
            if not os.path.exists(json_file_path):
                print("⚠️ No saved normal behavior found!")
                return
                
            try:
                with open(json_file_path, "r") as f:
                    # Read the content first to help with debugging
                    content = f.read()
                    try:
                        normal_behavior_list = json.loads(content)
                    except json.JSONDecodeError as json_err:
                        print(f"❌ JSON parsing error:")
                        print(f"  - Error message: {str(json_err)}")
                        print(f"  - Line number: {json_err.lineno}")
                        print(f"  - Column: {json_err.colno}")
                        # Print the content around the error location
                        error_pos = json_err.pos
                        start = max(0, error_pos - 50)
                        end = min(len(content), error_pos + 50)
                        print(f"  - Content around error (±50 chars):")
                        print(f"    {content[start:end]}")
                        raise
                        
                normal_behavior = np.array(normal_behavior_list)
                
                # Validate the data
                if normal_behavior.size == 0:
                    raise ValueError("Normal behavior array is empty")
                    
                # Initialize the Digital Twin with the loaded data
                self.digital_twin.learn_normal_behavior(normal_behavior)
                print(f"✅ Digital Twin initialized")
                
            except IOError as io_err:
                print(f"❌ Error reading file: {io_err}")
                raise
                
        except Exception as e:
            print(f"❌ Error initializing Digital Twin: {e}")
            raise


    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        print(f"[DEBUG] Aggregation started for round {rnd}")

        try:
            # Extract updates from results
            updates = []
            metrics = []
            
            for client_proxy, fit_res in results:
                updates.append(fit_res.parameters)
                if fit_res.metrics:
                    metrics.append((fit_res.num_examples, fit_res.metrics))
            
            # Validate updates using Digital Twin
            validated_updates = []
            for idx, update in enumerate(updates):
                update_ndarrays = parameters_to_ndarrays(update)
                
                # Extract features properly maintaining the dimension
                features = np.zeros(42)  # Initialize with correct size
                weights_mean = np.mean([np.mean(w) for w in update_ndarrays])
                weights_std = np.std([np.std(w) for w in update_ndarrays])
                
                # Fill the features array with statistics
                features[0] = weights_mean
                features[1] = weights_std
                # Remaining features can be filled with other statistics or zeros
                
                # Reshape features to match expected dimensions
                features = features.reshape(1, -1)  # Make it 2D: (1, 42)
                
                # Use the validate_data method from DigitalTwin class
                is_valid = self.digital_twin.validate_data(features)

                print(f"[Validation] Client {idx}:")
                print(f"- Mean: {weights_mean:.6f}")
                print(f"- Std: {weights_std:.6f}")
                print(f"- Valid: {is_valid}")

                if is_valid:
                    validated_updates.append(update)
                    print(f"✅ Update from client {idx} accepted")
                else:
                    print(f"❌ Update from client {idx} rejected")

            print(f"[DEBUG] Proceeding with {len(validated_updates)} validated results")

            # Update the Digital Twin's normal behavior with the validated updates
            if validated_updates:
                # Create aggregated features with correct dimensions
                aggregated_features = np.zeros((1, 42))  # Initialize with correct shape
                
                # Calculate aggregate statistics
                mean_weights = np.mean([
                    np.mean([np.mean(w) for w in parameters_to_ndarrays(update)])
                    for update in validated_updates
                ])
                std_weights = np.mean([
                    np.std([np.std(w) for w in parameters_to_ndarrays(update)])
                    for update in validated_updates
                ])
                
                # Fill aggregated features
                aggregated_features[0, 0] = mean_weights
                aggregated_features[0, 1] = std_weights
                # Remaining features stay as zeros
                
                try:
                    self.digital_twin.update_normal_behavior(aggregated_features)
                    print("✅ Digital Twin updated successfully")
                except Exception as e:
                    print(f"❌ Error updating normal behavior: {e}")

            # Proceed with FedAvg aggregation
            return super().aggregate_fit(rnd, results, failures)

        except Exception as e:
            print(f"❌ Error in aggregate_fit: {e}")
            raise



def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Calculate weighted average of metrics."""
    if not metrics:
        return {}
    
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    weighted_metrics = {}
    for metric_name in metrics[0][1].keys():
        weighted_sum = sum(value * num_examples for num_examples, m in metrics 
                         for name, value in m.items() if name == metric_name)
        weighted_metrics[metric_name] = weighted_sum / total_examples
    
    return weighted_metrics

def get_server_strategy():
    return CustomFedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

if __name__ == "__main__":
    start_time = time.time()
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Start the server
    try:
        history = fl.server.start_server(
            server_address="0.0.0.0:8601",
            strategy=get_server_strategy(),
            config=fl.server.ServerConfig(num_rounds=5)
        )
        print("✅ Server completed successfully!")
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise

     # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time

    # Prepare summary output
    summary_output = "[SUMMARY]\n"
    summary_output += f"INFO : Run finished 5 rounds in {total_time:.2f}s\n"

    # Extract metrics for each round
    loss_history = history.metrics_distributed["loss"]
    # binary_acc_history = history.metrics_distributed["binary_accuracy"]
    model_acc_history = history.metrics_distributed.get("accuracy", [])  # Add "accuracy" if available

    # Format metrics
    summary_output += "'LOSS':\n"
    for round_num, loss in loss_history:
        summary_output += f"Round {round_num}: {loss:.3f}\n"

    # summary_output += "\n'BINARY ACCURACY':\n"
    # for round_num, binary_acc in binary_acc_history:
    #     summary_output += f"Round {round_num}: {binary_acc:.3f}\n"

    if model_acc_history:
        summary_output += "\n'MODEL ACCURACY':\n"
        for round_num, model_acc in model_acc_history:
            summary_output += f"Round {round_num}: {model_acc:.3f}\n"

    # Directory path
    directory_path = r"C:/Users/ASWATHY.I/Desktop/MyFLIDS"

    # File name
    filename = "output.txt"

    # Full path to the output file
    output_file = os.path.join(directory_path, filename)
    plot_path = os.path.join(directory_path,"plot.png")

    try:
         # Write summary output to file
        with open(output_file, 'w') as f:
            f.write(summary_output)
        print("Summary output successfully written to file.")
    except Exception as e:
        print("Error occurred while writing summary output to file:", e)
    plot.plot_metrics(output_file, save_path=plot_path)

