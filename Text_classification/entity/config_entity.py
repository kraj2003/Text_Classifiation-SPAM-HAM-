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

