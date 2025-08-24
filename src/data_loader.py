"""
Data loader for SQuAD dataset and preprocessing utilities.

This module provides functionality to load, preprocess, and manage the SQuAD dataset
for Retrieval-Augmented Generation applications and drift detection testing.

Author: Cam
Date: July 2025
"""

import json
import random
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm


class SquadDataLoader:
    """Handles loading and preprocessing of SQuAD dataset for RAG applications."""
    
    def __init__(self, dataset_name: str = "squad", split: str = "train"):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the dataset (default: "squad")
            split: Dataset split to load (default: "train")
        """
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None
        self.processed_data = None
        
    def load_dataset(self) -> None:
        """Load the SQuAD dataset from HuggingFace."""
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"Loaded {len(self.dataset)} samples")
        
    def preprocess_data(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Preprocess the dataset into a structured format.
        
        Args:
            max_samples: Maximum number of samples to process (for faster testing)
            
        Returns:
            DataFrame with processed data
        """
        if self.dataset is None:
            self.load_dataset()
            
        data = []
        samples = self.dataset[:max_samples] if max_samples else self.dataset
        
        print("Preprocessing data...")
        for i, sample in enumerate(tqdm(samples)):
            # Debug: print the first sample to understand the structure
            if len(data) == 0:
                print(f"Sample type: {type(sample)}")
                print(f"Sample keys: {list(sample.keys()) if hasattr(sample, 'keys') else 'No keys'}")
                print(f"Sample: {sample}")
            
            # Extract the first answer if multiple exist
            answers = sample['answers']
            answer_text = answers['text'][0] if answers['text'] else ""
            answer_start = answers['answer_start'][0] if answers['answer_start'] else -1
            
            data.append({
                'id': sample['id'],
                'title': sample['title'],
                'context': sample['context'],
                'question': sample['question'],
                'answer_text': answer_text,
                'answer_start': answer_start,
                'is_impossible': sample.get('is_impossible', False)
            })
            
        self.processed_data = pd.DataFrame(data)
        print(f"Preprocessed {len(self.processed_data)} samples")
        return self.processed_data
    
    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the processed data into train and test sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.processed_data is None:
            self.preprocess_data()
            
        # Remove samples with empty answers
        valid_data = self.processed_data[self.processed_data['answer_text'].str.len() > 0].copy()
        
        # Split the data
        test_size_int = int(len(valid_data) * test_size)
        test_indices = random.sample(range(len(valid_data)), test_size_int)
        train_indices = [i for i in range(len(valid_data)) if i not in test_indices]
        
        train_data = valid_data.iloc[train_indices].reset_index(drop=True)
        test_data = valid_data.iloc[test_indices].reset_index(drop=True)
        
        print(f"Train set: {len(train_data)} samples")
        print(f"Test set: {len(test_data)} samples")
        
        return train_data, test_data
    
    def create_knowledge_base(self, data: pd.DataFrame) -> List[Dict]:
        """
        Create a knowledge base from the dataset.
        
        Args:
            data: DataFrame with processed data
            
        Returns:
            List of knowledge base entries
        """
        knowledge_base = []
        
        for _, row in data.iterrows():
            knowledge_base.append({
                'id': row['id'],
                'title': row['title'],
                'content': row['context'],
                'question': row['question'],
                'answer': row['answer_text']
            })
            
        return knowledge_base
    
    def save_processed_data(self, filepath: str) -> None:
        """Save processed data to JSON file."""
        if self.processed_data is not None:
            self.processed_data.to_json(filepath, orient='records', indent=2)
            print(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filepath: str) -> pd.DataFrame:
        """Load processed data from JSON file."""
        self.processed_data = pd.read_json(filepath)
        print(f"Loaded processed data from {filepath}")
        return self.processed_data


def create_drift_scenarios(base_data: pd.DataFrame, num_scenarios: int = 4) -> List[pd.DataFrame]:
    """
    Create different drift scenarios by modifying the base dataset.
    
    Args:
        base_data: Original dataset
        num_scenarios: Number of drift scenarios to create
        
    Returns:
        List of modified datasets representing different drift scenarios
    """
    scenarios = []
    
    for i in range(num_scenarios):
        scenario_data = base_data.copy()
        
        if i == 0:
            # Scenario 1: Add new content (10% new samples)
            new_samples = int(len(base_data) * 0.1)
            new_data = base_data.sample(n=new_samples, random_state=i).copy()
            new_data['context'] = new_data['context'].apply(lambda x: x + " [UPDATED]")
            new_data['id'] = new_data['id'].apply(lambda x: f"{x}_new_{i}")
            scenario_data = pd.concat([scenario_data, new_data], ignore_index=True)
            
        elif i == 1:
            # Scenario 2: Remove some content (5% removal)
            remove_samples = int(len(base_data) * 0.05)
            indices_to_remove = random.sample(range(len(base_data)), remove_samples)
            scenario_data = scenario_data.drop(indices_to_remove).reset_index(drop=True)
            
        elif i == 2:
            # Scenario 3: Modify existing content (15% modifications)
            modify_samples = int(len(base_data) * 0.15)
            indices_to_modify = random.sample(range(len(base_data)), modify_samples)
            for idx in indices_to_modify:
                scenario_data.loc[idx, 'context'] = scenario_data.loc[idx, 'context'] + " [MODIFIED]"
                
        elif i == 3:
            # Scenario 4: Add noise (unanswerable questions)
            noise_samples = int(len(base_data) * 0.1)
            noise_data = base_data.sample(n=noise_samples, random_state=i).copy()
            noise_data['answer_text'] = ""
            noise_data['answer_start'] = -1
            noise_data['is_impossible'] = True
            noise_data['id'] = noise_data['id'].apply(lambda x: f"{x}_noise_{i}")
            scenario_data = pd.concat([scenario_data, noise_data], ignore_index=True)
        
        scenarios.append(scenario_data)
        print(f"Created scenario {i+1}: {len(scenario_data)} samples")
    
    return scenarios


if __name__ == "__main__":
    # Example usage
    loader = SquadDataLoader()
    train_data, test_data = loader.split_train_test()
    
    # Save processed data
    loader.save_processed_data("data/processed_squad_train.json")
    
    # Create drift scenarios
    scenarios = create_drift_scenarios(train_data)
    
    for i, scenario in enumerate(scenarios):
        scenario.to_json(f"data/scenario_{i+1}.json", orient='records', indent=2)
