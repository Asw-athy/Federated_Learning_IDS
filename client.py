import os
import numpy as np
import flwr as fl
import logging
import json
from loader import DataLoader, ModelLoader
# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.ensemble import IsolationForest

class DigitalTwin:
    """Simulates and validates client behavior dynamically."""

    def __init__(self, client_id, threshold_factor=2.0):
        """
        Initialize the Digital Twin with adaptive anomaly detection.

        :param client_id: Identifier for the client.
        :param threshold_factor: Multiplier for standard deviation to determine anomalies.
        """
        self.client_id = client_id
        self.threshold_factor = threshold_factor  # Controls anomaly sensitivity
        self.normal_behavior = None
        self.threshold = None
        self.anomaly_detector = None  # Placeholder for Isolation Forest

    def learn_normal_behavior(self, data):
        """
        Learn normal behavior from actual training data.

        :param data: The dataset used to define normal behavior.
        """
        self.normal_behavior = data
        mean = np.mean(data)
        std_dev = np.std(data)
        self.threshold = mean + self.threshold_factor * std_dev  # Dynamic threshold

        # Train Isolation Forest for better anomaly detection
        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.anomaly_detector.fit(data.reshape(-1, 1))

        print(f"✅ Digital Twin trained: Mean={mean}, Std Dev={std_dev}, Threshold={self.threshold}")

    def validate_data(self, data):
        """Validate if the data follows normal behavior."""
        try:
            # Ensure data is 2D
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            mean_data = np.mean(data)
            mean_behavior = np.mean(self.normal_behavior)
            
            deviation = abs(mean_data - mean_behavior)
            is_valid = deviation <= self.threshold
            
            feedback = {
                "mean_data": mean_data,
                "mean_behavior": mean_behavior,
                "deviation": deviation,
                "threshold": self.threshold,
                "is_valid": is_valid
            }
            
            logging.debug(f"Data validation feedback: {feedback}")
            return is_valid
            
        except Exception as e:
            print(f"❌ Error in validate_data: {e}")
            return False


    def update_normal_behavior(self, new_data):
        """Update the normal behavior with new data."""
        try:
            # Ensure new_data is 2D with correct shape
            if new_data.ndim == 1:
                new_data = new_data.reshape(1, -1)
            
            # Ensure new_data has correct number of features
            if new_data.shape[1] != 42:
                raise ValueError(f"Expected 42 features, got {new_data.shape[1]}")
            
            # Ensure self.normal_behavior is 2D
            if self.normal_behavior.ndim == 1:
                self.normal_behavior = self.normal_behavior.reshape(1, -1)
                
            # Now concatenate
            all_data = np.concatenate((self.normal_behavior, new_data), axis=0)
            
            # Update normal behavior
            self.normal_behavior = all_data
            self.mean = np.mean(self.normal_behavior, axis=0)
            print(f"✅ Normal behavior updated.")
            
            # Save the updated normal behavior
            try:
                normal_behavior_list = self.normal_behavior.tolist()
                with open("saved_model_fixed.json", "w") as f:
                    json.dump(normal_behavior_list, f)
                print("💾 Normal behavior saved successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not save normal behavior: {e}")
                
        except Exception as e:
            print(f"❌ Error updating normal behavior: {e}")
            raise

class Client(fl.client.NumPyClient):
    def __init__(self):
        data_loader = DataLoader("data/balanced_training_set.csv", "data/balanced_testing_set.csv")
        self.X_train, self.Y_train, self.X_test, self.Y_test = data_loader.get_data()
        self.model = ModelLoader.get_model(self.X_train.shape[1:])
        print(self.X_train.shape)  # Debug to get the correct input shape
        self.digital_twin = DigitalTwin(client_id="client1")

        import json

        # Train Digital Twin with actual training data
        self.digital_twin.learn_normal_behavior(self.X_train)
        print(f"✅ Normal behavior shape: {self.digital_twin.normal_behavior.shape}")
        print(f"✅ Threshold computed: {self.digital_twin.threshold}")


        import numpy as np

        # Convert NumPy array to a Python list before saving
        with open("saved_model_fixed.json", "w") as f:
            json.dump(self.digital_twin.normal_behavior.tolist(), f)

        print("💾 Digital Twin normal behavior saved.")


        # Train Digital Twin with actual training data
        self.digital_twin.learn_normal_behavior(self.X_train)
        
        # Load saved model if exists
        if os.path.exists("saved_model.h5"):
            self.model.load_weights("saved_model.h5")
            print("✅ Loaded saved model weights.")
        
    def get_parameters(self, config):
        return self.model.get_weights()    

    def fit(self, parameters, _):
        is_valid = self.digital_twin.validate_data(self.X_train)
        if not is_valid:
            print("🚨 Data validation failed. Skipping training.")
            return self.model.get_weights(), len(self.X_train), {}

        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.Y_train, epochs=1, batch_size=64)

        # ✅ Update Digital Twin with new behavior
        self.digital_twin.update_normal_behavior(self.X_train)

        # ✅ Save the trained model to disk with error handling
        try:
            # Generate a unique filename using timestamp
            import time
            timestamp = int(time.time())
            model_filename = f"saved_model_{timestamp}.h5"
            
            # Try to save with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model.save(model_filename)
                    print(f"💾 Model saved after training as {model_filename}")
                    
                    # If successful, try to rename to the final filename
                    if os.path.exists("saved_model.h5"):
                        os.remove("saved_model.h5")
                    os.rename(model_filename, "saved_model.h5")
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Retry {attempt + 1} of {max_retries} for saving model...")
                        time.sleep(1)  # Wait before retry
                    else:
                        print(f"❌ Failed to save model after {max_retries} attempts: {e}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            # Continue execution even if save fails
            pass

        # Prepare metrics
        history_metrics = {
            k: float(v[-1]) if isinstance(v, (list, np.ndarray)) and len(v) > 0 else float(v)
            for k, v in history.history.items() if v
        }

        digital_twin_metric = float(np.mean(self.digital_twin.normal_behavior)) if self.digital_twin.normal_behavior.size > 0 else 0.0

        return self.model.get_weights(), len(self.X_train), {**history_metrics, "digital_twin": digital_twin_metric}

    
    def evaluate(self, parameters, _):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"loss": loss, "accuracy": accuracy}


if __name__ == "__main__":
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8601")
    fl.client.start_numpy_client(server_address=server_address, client=Client())

