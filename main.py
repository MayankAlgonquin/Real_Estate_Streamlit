# from setuptools import find_packages, setup


from src.data.make_dataset import load_and_clean_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance, plot_confusion_matrix, plot_actual_vs_predicted
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model
import logging
import os
#Implementing logger to track the main pipeline.
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
if __name__ == "__main__":
    # Load and preprocess the data
    try:
        data_path = "data/raw/real_estate.csv"
        df =  load_and_clean_data(data_path)
    except FileNotFoundError:
        logger.error("File not found. Please enter a valid file.")
        
    except Exception as e:
        logger.exception(f"Error while loading data: {e}")


    # Create dummy variables and separate features and target
    try:
        X, y = create_dummy_vars(df)
        logger.info("Feature engineering completed")
    except Exception as e:
        logger.exception(f"Error in creating dummy variables: {e}")

    # Train the linear regression model
    try:
        model, X_test_scaled, y_test = train_RFmodel(X, y)
        logger.info("Model trained successfully")
    except Exception as e:
        logger.exception(f"Error during model training: {e}")


    # Evaluate the model
    try:
        plot_feature_importance(model, X)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
        plot_actual_vs_predicted(y_test, y_pred)
        logger.info("Model evaluated successfully")
    except Exception as e:
        logger.exception(f"Error during evaluation: {e}")

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
