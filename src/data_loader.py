import json
import os
from typing import Dict
from src.models import DatasetModel


class DatasetLoader:
    """Singleton service to load, validate, and manage domain JSON datasets."""

    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory
        self.datasets: Dict[str, DatasetModel] = {}
        self.load_all_datasets()

    def load_all_datasets(self):
        """Discovers and parses all JSON datasets from the data directory."""
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Data directory '{self.data_directory}' does not exist.")

        for filename in os.listdir(self.data_directory):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_directory, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    dataset = DatasetModel(**raw_data)
                    self.datasets[dataset.dataset_id] = dataset
                    print(f"[DataLoader] Successfully loaded dataset: '{dataset.dataset_id}' ({len(dataset.nodes)} nodes)")

    def get_dataset(self, dataset_id: str) -> DatasetModel:
        """Retrieves a specific dataset by its unique identifier."""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found. Available: {list(self.datasets.keys())}")
        return self.datasets[dataset_id]

    def get_available_datasets(self) -> Dict[str, str]:
        """Returns mapping of dataset IDs to human-readable names for frontend dropdown."""
        return {d_id: d.name for d_id, d in self.datasets.items()}


# Global singleton instance for application-wide reuse
data_loader = DatasetLoader()