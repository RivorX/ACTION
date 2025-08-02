import pandas as pd
import logging
from scripts.utils.config_manager import ConfigManager
from scripts.utils.preprocessing_utils import PreprocessingUtils
from pytorch_forecasting import TimeSeriesDataSet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Klasa odpowiedzialna za preprocessing danych giełdowych i tworzenie zbioru danych TimeSeriesDataSet."""
    
    def __init__(self, config: dict):
        self.config = config
        self.preprocessing_utils = PreprocessingUtils(config)

    def preprocess_data(self, df: pd.DataFrame) -> TimeSeriesDataSet:
        df, _ = self.preprocessing_utils.preprocess_dataframe(df)
        return self.preprocessing_utils.create_dataset(df)