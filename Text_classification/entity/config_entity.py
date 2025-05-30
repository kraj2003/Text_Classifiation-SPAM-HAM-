from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_file_path: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE:Path
    data_dir:Path
    all_schema: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    final_model: Path
    preprocessor: Path

@dataclass
class ModelTrainerConfig:
    root_dir:Path
    test_data:Path
    train_data:Path

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    mlflow_uri: str
