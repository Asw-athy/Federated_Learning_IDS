import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from digital_twin import DigitalTwin
import typing
import logging

class DataLoader:
    def __init__(self, train_file_path: str, test_file_path: str):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.digital_twin = DigitalTwin(client_id="DataLoader")
        self.label_encoders = {}
        self.scaler = None
        logging.basicConfig(level=logging.DEBUG)  # Enable logging at DEBUG level
    
    def load_data(self, file_path: str):
        data = pd.read_csv(file_path)
        drop_columns = [col for col in ["id", "attack_cat"] if col in data.columns]
        data = data.drop(columns=drop_columns, errors='ignore')
        return data

    def preprocess_data(self, train_data, test_data):
        categorical_columns = train_data.select_dtypes(include=['object']).columns
        numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
        
        # Fit transformers on training data
        self.label_encoders = {}
        for col in categorical_columns:
            encoder = preprocessing.LabelEncoder()
            # Fit encoder on both train and test data (combine data to fit on all possible labels)
            combined_data = pd.concat([train_data[col], test_data[col]], axis=0)
            encoder.fit(combined_data)  # Fit on both training and test data
            self.label_encoders[col] = encoder
        
        self.scaler = preprocessing.MinMaxScaler().fit(train_data[numerical_columns])
        
        # Apply transformations
        for col in categorical_columns:
            # Transform train data
            train_data[col] = self.label_encoders[col].transform(train_data[col])
            
            # Handle unseen categories in test data (if any) by mapping them to 'UNKNOWN'
            test_data[col] = test_data[col].apply(
                lambda x: x if x in self.label_encoders[col].classes_ else 'UNKNOWN'
            )
            # Transform the 'UNKNOWN' category to a valid value
            test_data[col] = self.label_encoders[col].transform(test_data[col])
        
        train_data[numerical_columns] = self.scaler.transform(train_data[numerical_columns])
        test_data[numerical_columns] = self.scaler.transform(test_data[numerical_columns])

        # Separate features and labels for both train and test data
        X_train, Y_train = self.separate_features_and_labels(train_data)
        X_test, Y_test = self.separate_features_and_labels(test_data)
        
        return X_train, Y_train, X_test, Y_test
    
    def validate_with_digital_twin(self, data):
        mean_data = np.mean(data.to_numpy())
        mean_behavior = np.mean(self.digital_twin.normal_behavior)
        deviation = mean_data - mean_behavior
        is_valid = abs(deviation) <= self.digital_twin.threshold
        
        # Detailed feedback for debugging
        feedback = {
            "mean_data": mean_data,
            "mean_behavior": mean_behavior,
            "deviation": deviation,
            "threshold": self.digital_twin.threshold,
            "is_valid": is_valid
        }

        logging.debug(f"Data validation feedback: {feedback}")
        
        return is_valid, feedback
    
    def align_digital_twin(self, data):
        self.digital_twin.update_normal_behavior(data.to_numpy())

    def separate_features_and_labels(self, data):
        Y = data.label.to_numpy()
        X = data.drop(columns="label").to_numpy()
        return X, Y
    
    def get_data(self):
        if not self.train_file_path or not self.test_file_path:
            raise ValueError("File paths for training and testing datasets must be provided.")

        train_data = self.load_data(self.train_file_path)
        test_data = self.load_data(self.test_file_path)

        # Preprocess the data
        X_train, Y_train, X_test, Y_test = self.preprocess_data(train_data, test_data)

        # Align Digital Twin if deviation is too large
        self.align_digital_twin(train_data)

        # Validate datasets
        is_train_valid, feedback_train = self.validate_with_digital_twin(train_data)
        is_test_valid, feedback_test = self.validate_with_digital_twin(test_data)

        # Log the validation results
        logging.debug(f"Train validation result: {is_train_valid}, Feedback: {feedback_train}")
        logging.debug(f"Test validation result: {is_test_valid}, Feedback: {feedback_test}")

        if not is_train_valid:
            logging.error(f"Train data validation failed! Details: {feedback_train}")
            raise ValueError("Invalid training data detected by Digital Twin!")

        if not is_test_valid:
            logging.error(f"Test data validation failed! Details: {feedback_test}")
            raise ValueError("Invalid testing data detected by Digital Twin!")

        return X_train, Y_train, X_test, Y_test

class ModelLoader:
    @staticmethod
    def get_model(sample_shape: typing.Tuple[int]) -> tf.keras.Model:
        import tensorflow as tf
        inputs = tf.keras.Input(sample_shape)
        x = tf.keras.layers.Dense(100, activation="relu")(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(50, activation="relu")(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model
