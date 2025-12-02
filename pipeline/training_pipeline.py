from src.data_processing import DataProcessing
from src.model_training import ModelTraining

if __name__=="__main__":
    data_processing = DataProcessing("artifacts/raw/data.csv")
    data_processing.run()

    model_training = ModelTraining()
    model_training.run()