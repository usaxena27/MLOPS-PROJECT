import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self):
        self.processed_data_path = "artifacts/processed"
        self.model_path = "artifacts/model"
        os.makedirs(self.model_path, exist_ok=True)
        self.model = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=42)
        logger.info("Model Training Initialized")

    def load_data(self):
        try:
            X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            Y_train = joblib.load(os.path.join(self.processed_data_path, "Y_train.pkl"))
            Y_test = joblib.load(os.path.join(self.processed_data_path, "Y_test.pkl"))

            logger.info("Data Loaded Successful")
            return X_train, X_test, Y_train, Y_test
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)

    def train_model(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, os.path.join(self.model_path, "model.pkl"))
            logger.info("Model trained and saved successfully")

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise CustomException("Failed to train model", e)

    def evaluate_model(self, X_test, Y_test):   # <-- keep Y_test to match run()
        try:
            # use the arguments, not self.X_test / self.Y_test
            Y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred, average='weighted')
            recall = recall_score(Y_test, Y_pred, average='weighted')
            f1 = f1_score(Y_test, Y_pred, average='weighted')
            conf_matrix = confusion_matrix(Y_test, Y_pred)

            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_matrix,
                annot=True,
                cmap='Blues',
                xticklabels=np.unique(Y_test),
                yticklabels=np.unique(Y_test)
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            conf_matrix_path = os.path.join(self.model_path, "confusion_matrix.png")
            plt.savefig(conf_matrix_path)
            plt.close()

            logger.info(f"Confusion Matrix saved successfully at {conf_matrix_path}")

        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)

    def run(self):
        try:
            X_train, X_test, Y_train, Y_test = self.load_data()
            self.train_model(X_train, Y_train)
            self.evaluate_model(X_test, Y_test)

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise CustomException("Failed to run model training", e)
        
if __name__ == "__main__":
    model_training = ModelTraining()
    model_training.run()
