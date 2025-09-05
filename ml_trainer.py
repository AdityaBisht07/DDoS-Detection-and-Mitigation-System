#!/usr/bin/env python3
"""
DDoS Detection ML Trainer
========================

Machine learning model training for DDoS attack detection.
Generates synthetic training data and trains various ML models.

Author: DDoS Detection System
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DDoSTrainingDataGenerator:
    """Generate synthetic training data for DDoS detection."""
    
    def __init__(self):
        self.normal_traffic_params = {
            'packet_rate': (10, 100),
            'unique_sources': (5, 50),
            'syn_ratio': (0.1, 0.3),
            'udp_ratio': (0.1, 0.4),
            'http_ratio': (0.2, 0.6),
            'avg_packet_size': (500, 1500),
            'connection_attempts': (5, 30),
            'established_connections': (3, 25)
        }
        
        self.attack_patterns = {
            'syn_flood': {
                'packet_rate': (1000, 10000),
                'unique_sources': (1, 10),
                'syn_ratio': (0.8, 1.0),
                'udp_ratio': (0.0, 0.1),
                'http_ratio': (0.0, 0.1),
                'avg_packet_size': (40, 60),
                'connection_attempts': (500, 5000),
                'established_connections': (0, 50)
            },
            'udp_flood': {
                'packet_rate': (2000, 20000),
                'unique_sources': (1, 20),
                'syn_ratio': (0.0, 0.1),
                'udp_ratio': (0.9, 1.0),
                'http_ratio': (0.0, 0.1),
                'avg_packet_size': (100, 200),
                'connection_attempts': (100, 1000),
                'established_connections': (0, 10)
            },
            'http_flood': {
                'packet_rate': (500, 5000),
                'unique_sources': (1, 5),
                'syn_ratio': (0.1, 0.3),
                'udp_ratio': (0.0, 0.1),
                'http_ratio': (0.8, 1.0),
                'avg_packet_size': (800, 2000),
                'connection_attempts': (200, 2000),
                'established_connections': (50, 200)
            }
        }
    
    def generate_normal_traffic(self, n_samples: int = 1000) -> List[Dict]:
        """Generate normal traffic samples."""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for feature, (min_val, max_val) in self.normal_traffic_params.items():
                if isinstance(min_val, int):
                    sample[feature] = random.randint(min_val, max_val)
                else:
                    sample[feature] = random.uniform(min_val, max_val)
            
            sample['is_attack'] = 0
            sample['attack_type'] = 'normal'
            samples.append(sample)
        
        return samples
    
    def generate_attack_traffic(self, attack_type: str, n_samples: int = 500) -> List[Dict]:
        """Generate attack traffic samples."""
        if attack_type not in self.attack_patterns:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        samples = []
        params = self.attack_patterns[attack_type]
        
        for _ in range(n_samples):
            sample = {}
            for feature, (min_val, max_val) in params.items():
                if isinstance(min_val, int):
                    sample[feature] = random.randint(min_val, max_val)
                else:
                    sample[feature] = random.uniform(min_val, max_val)
            
            sample['is_attack'] = 1
            sample['attack_type'] = attack_type
            samples.append(sample)
        
        return samples
    
    def generate_dataset(self, normal_samples: int = 2000, attack_samples_per_type: int = 500) -> pd.DataFrame:
        """Generate complete training dataset."""
        logger.info("Generating training dataset...")
        
        # Generate normal traffic
        normal_data = self.generate_normal_traffic(normal_samples)
        
        # Generate attack traffic
        attack_data = []
        for attack_type in self.attack_patterns.keys():
            attack_data.extend(self.generate_attack_traffic(attack_type, attack_samples_per_type))
        
        # Combine and shuffle
        all_data = normal_data + attack_data
        random.shuffle(all_data)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated dataset with {len(df)} samples")
        logger.info(f"Normal samples: {len(df[df['is_attack'] == 0])}")
        logger.info(f"Attack samples: {len(df[df['is_attack'] == 1])}")
        
        return df

class DDoSMLTrainer:
    """Machine learning trainer for DDoS detection."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'packet_rate', 'unique_sources', 'syn_ratio', 'udp_ratio', 
            'http_ratio', 'avg_packet_size', 'connection_attempts', 
            'established_connections'
        ]
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        X = df[self.feature_columns].values
        y = df['is_attack'].values
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train multiple ML models."""
        logger.info("Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Define models
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_config.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test_scaled, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
        
        self.models = {name: results[name]['model'] for name in results.keys()}
        
        return results
    
    def save_models(self, filepath: str = "models/"):
        """Save trained models and scalers."""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{filepath}/{name}_model.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{filepath}/{name}_scaler.pkl")
        
        # Save feature columns
        with open(f"{filepath}/feature_columns.json", 'w') as f:
            json.dump(self.feature_columns, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = "models/"):
        """Load trained models and scalers."""
        import os
        
        if not os.path.exists(filepath):
            logger.warning(f"Model directory {filepath} not found")
            return False
        
        try:
            # Load models
            for name in ['random_forest', 'gradient_boosting', 'svm', 'neural_network']:
                model_path = f"{filepath}/{name}_model.pkl"
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            # Load scalers
            scaler_path = f"{filepath}/main_scaler.pkl"
            if os.path.exists(scaler_path):
                self.scalers['main'] = joblib.load(scaler_path)
            
            # Load feature columns
            features_path = f"{filepath}/feature_columns.json"
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, features: Dict) -> Dict:
        """Make prediction using trained models."""
        if not self.models or 'main' not in self.scalers:
            logger.warning("Models not loaded")
            return {'error': 'Models not loaded'}
        
        # Prepare features
        feature_vector = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
        feature_vector_scaled = self.scalers['main'].transform(feature_vector)
        
        predictions = {}
        
        for name, model in self.models.items():
            try:
                prediction = model.predict(feature_vector_scaled)[0]
                probability = model.predict_proba(feature_vector_scaled)[0][1]
                
                predictions[name] = {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'confidence': float(probability) if prediction == 1 else float(1 - probability)
                }
            except Exception as e:
                logger.error(f"Error with {name} model: {e}")
                predictions[name] = {'error': str(e)}
        
        return predictions

def main():
    """Main function to train models."""
    print("🤖 DDoS Detection ML Trainer")
    print("=" * 40)
    
    # Generate training data
    generator = DDoSTrainingDataGenerator()
    df = generator.generate_dataset(normal_samples=2000, attack_samples_per_type=500)
    
    # Train models
    trainer = DDoSMLTrainer()
    X, y = trainer.prepare_data(df)
    results = trainer.train_models(X, y)
    
    # Print results
    print("\n📊 Model Performance:")
    print("-" * 30)
    for name, result in results.items():
        print(f"{name:20} | Accuracy: {result['accuracy']:.3f} | AUC: {result['auc_score']:.3f}")
    
    # Save models
    trainer.save_models()
    
    print("\n✅ Training complete! Models saved to 'models/' directory")
    print("🚀 You can now use these models in your DDoS detection system")

if __name__ == "__main__":
    main()
