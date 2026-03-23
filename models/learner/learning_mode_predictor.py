"""
LSTM Model for Predicting Learning Mode Recommendation
Trained on user interaction logs in VLEs to suggest resources based on previous actions.

-archive 
-offer adaptive resources
-sprint mode
-todays recommendation
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tfact
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class LearningModePredictor:
    """LSTM model for predicting recommended learning mode based on user activity"""
    
    def __init__(self, sequence_length=10, embedding_dim=32, lstm_units=128, no_of_files = 20,
                    
                    data_dir='data/training_files/KT4Subset'):
            """
            Initialize the LSTM model
            
            Args:
                sequence_length: Number of previous actions to consider
                embedding_dim: Dimension for action embedding
                lstm_units: Number of LSTM units
                data_dir: Directory containing training data
            """
            self.sequence_length = sequence_length
            self.embedding_dim = embedding_dim
            self.lstm_units = lstm_units
            self.data_dir = data_dir
            self.model = None
            self.no_of_files = no_of_files
            
            # Label encoders
            self.action_encoder = LabelEncoder()
            self.item_type_encoder = LabelEncoder()
            self.mode_encoder = LabelEncoder()
            
            # Define learning modes
            self.learning_modes = ['sprint', 'assignment','instructional_material',
                                   
                                    'adaptive_offer', 'todays_recommendation']
        
    def load_data(self, data_dir=None):
        """
        Load and preprocess all user CSV files
        
        Args:
            data_dir: Directory containing user CSV files
            If None, uses the default directory set during initialization.
        Returns:
            DataFrame with all user data
        """
        if data_dir is None:
            data_dir = self.data_dir
        data_path = Path(data_dir)
        all_data = []
        
        # Load all user files
        for i, csv_file in enumerate(data_path.glob('u*.csv')):
            if i >= self.no_of_files:
                break
            if csv_file.suffix == '.csv' and not csv_file.name.endswith(':Zone.Identifier'):
                df = pd.read_csv(csv_file)
                user_id = csv_file.stem  # e.g., 'u20'
                df['user_id'] = user_id
                all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(all_data)} user files with {len(combined_df)} total interactions")
        
        return combined_df
    
    def map_source_to_mode(self, source):
        """
        Map source field to learning mode
        
        Args:
            source: Original source value from data
            
        Returns:
            Learning mode category
        """
        source = str(source).lower()
        
        # Map based on source patterns
        # The data souce uses more specific terms for each action
        # 
        if 'sprint' in source:
            return 'sprint'
        elif 'diagnosis' in source or 'review_quiz' in source:
            ##
            return 'assignment'
        elif 'archive' in source or 'library' in source:
            return 'instructional_material'
        elif 'adaptive_offer' in source or 'recommendation' in source:
            return 'adaptive_offer'
        else:
            return 'todays_recommendation'
    
    def preprocess_data(self, df):
        
        ## Only export learning related actions. Bundles, explanations, questions, lectures
        df = df[df['item_id'].str.contains('|'.join(['b', 'e', 'q', 'l']))]
        print(df)
        # Encode action types
        df['action_encoded'] = self.action_encoder.fit_transform(df['action_type'])
        
        # Map Item ID to appropriate Item Type
        # Only extract b, e, q, l types to exclude non learning actions.
        df['item_id'] = df['item_id'].astype(str).str[0]
        df['item_type'] = df['item_id'].astype(str).str[0]
        df['item_type'] = df['item_type'].replace({'b': 'bundle', 'e': 'explanation', 
                                                    'q': 'question', 'l': 'lecture'})

        df['item_type_encoded'] = self.item_type_encoder.fit_transform(df['item_type'])
        
        # Map source to learning mode
        df['learning_mode'] = df['source'].apply(self.map_source_to_mode)
        df['mode_encoded'] = self.mode_encoder.fit_transform(df['learning_mode'])
        
        # Calculate time difference between actions (in seconds)
        df['time_diff'] = df.groupby('user_id')['timestamp'].diff().fillna(0) / 1000
        
        # Check how often user responds consecutively to check for multiple attempts
        # indicates frustration
        df['is_respond'] = (df['action_type'] == 'respond').astype(int)
        df['consecutive_responds'] = df.groupby('user_id')['is_respond'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
        )
        
        self.num_actions = len(self.action_encoder.classes_)
        self.num_item_types = len(self.item_type_encoder.classes_)
        self.num_modes = len(self.mode_encoder.classes_)
        
        print(f"\nAction types: {list(self.action_encoder.classes_)}")
        print(f"Number of unique actions: {self.num_actions}")
        print(f"\nItem types: {list(self.item_type_encoder.classes_)}")
        print(f"Number of unique item types: {self.num_item_types}")
        print(f"\nLearning modes: {list(self.mode_encoder.classes_)}")
        print(f"Number of learning modes: {self.num_modes}")
        
        # Distribution of learning modes
        print(f"\nLearning mode distribution:")
        print(df['learning_mode'].value_counts())
        
        return df
    
    def create_sequences(self, df):
        """
        Create sequences for LSTM training
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            X, y: Input sequences (action_types) and target learning modes
        """
        sequences = []
        targets = []
        
        # Group by user to maintain sequence continuity

        #### LOOK INTO ADDING EXTRA SEQUENCE GROUPING BASED ON WHEN THE USER SWITCHES TO A NEW LEARNING MODE. EG. Repeating responses might indicate retries based on the item.

        for user_id in df['user_id'].unique():
            user_df = df[df['user_id'] == user_id].sort_values('timestamp')
            
            # Extract action and item_type encodings
            action_sequence = user_df['action_encoded'].values
            item_type_sequence = user_df['item_type_encoded'].values
            mode_sequence = user_df['mode_encoded'].values
            
            # Create sequences
            for i in range(len(action_sequence) - self.sequence_length):
                # Input: sequence of [action_type, item_type] pairs
                action_seq = action_sequence[i:i + self.sequence_length]
                item_type_seq = item_type_sequence[i:i + self.sequence_length]
                
                # Stack features: shape (sequence_length, 2)
                seq = np.column_stack([action_seq, item_type_seq])
                
                # Target: learning mode for the next interaction
                target = mode_sequence[i + self.sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        X = np.array(sequences)
        y = np.array(targets)
        
        print(f"\nCreated {len(X)} sequences of length {self.sequence_length}")
        print(f"Input shape: {X.shape} (sequence_length, features)")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def build_model(self, num_actions, num_item_types, num_modes):
        """
        Build the LSTM model architecture with direct feature input
        """
        model = Sequential([
            # Input shape: (sequence_length, 2) where 2 = [action_encoded, item_type_encoded]
            # First LSTM layer with return sequences
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, 2)),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            
            # Output layer for learning modes
            Dense(num_modes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        model.summary()
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the LSTM model

        """
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=self.num_modes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        

        if self.model is None:
            self.build_model(num_actions=self.num_actions, 
                           num_item_types=self.num_item_types,
                           num_modes=self.num_modes)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ModelCheckpoint('models/learning_mode_lstm_best.keras', 
                          monitor='val_accuracy', 
                          save_best_only=True, 
                          mode='max')
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_learning_mode(self, action_sequence, item_type_sequence):
        """
        Predict the recommended learning actions given the last actions of the user and the types of items they interacted with.
    
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        # Combine features: shape (1, sequence_length, 2)
        seq = np.array([np.column_stack([action_sequence, item_type_sequence])])
        # Predict
        prediction = self.model.predict(seq, verbose=0)
        print(prediction)
        predicted_class = np.argmax(prediction[0])
        predicted_mode = self.mode_encoder.inverse_transform([predicted_class])[0]
        confidence = prediction[0][predicted_class]
        
        # Get all predictions sorted by probability
        all_indices = np.argsort(prediction[0])[::-1]
        all_modes = self.mode_encoder.inverse_transform(all_indices)
        all_probs = prediction[0][all_indices]
        
        return {
            'predicted_mode': predicted_mode,
            'confidence': float(confidence),
            'all_predictions': [
                {'mode': mode, 'probability': float(prob)}
                for mode, prob in zip(all_modes, all_probs)
            ]
        }
    
    def predict_from_action_names(self, action_history= [[str,str]]):
        """
        Predict learning mode from lists of action types and item types
        
        Args:
            action_history: List of sets of action type strings (e.g., [{'enter','q'}, {'play_audio','l'}, ...])
            
        Returns:
            Prediction results
        """
        # Encode action names
        actions = []
        item_types = []
        for action_set in action_history:
            # Take first action in the set for encoding
            action = action_set[0]
            actions.append(action)
            # Infer item type from action if possible   
            if 'l' in action_set[1]:
                item_types.append('lecture')
            elif 'q' in action_set[1]:
                item_types.append('question')
            elif 'b' in action_set[1]:
                item_types.append('bundle')
            elif 'e' in action_set[1]:
                item_types.append('explanation')
            else:
                item_types.append('unknown')
        try:
            action_encoded = self.action_encoder.transform(actions)
        except ValueError as e:
            raise ValueError(f"Unknown action type in sequence: {e}")
 
        # Encode item types
        try:
            item_type_encoded = self.item_type_encoder.transform(item_types)
        except ValueError as e:
            raise ValueError(f"Unknown item type in sequence: {e}")
        
        # Pad or truncate to sequence length
        if len(action_encoded) < self.sequence_length:
            # Pad with zeros
            pad_length = self.sequence_length - len(action_encoded)
            action_encoded = np.pad(action_encoded, (pad_length, 0), mode='constant')
            item_type_encoded = np.pad(item_type_encoded, (pad_length, 0), mode='constant')
        elif len(action_encoded) > self.sequence_length:
            # Take last sequence_length actions
            action_encoded = action_encoded[-self.sequence_length:]
            item_type_encoded = item_type_encoded[-self.sequence_length:]
        
        return self.predict_learning_mode(action_encoded, item_type_encoded)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Loss and accuracy
        """
        y_test_categorical = to_categorical(y_test, num_classes=self.num_modes)
        loss, accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
        
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Detailed per-class accuracy
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print("\nPer-mode accuracy:")
        for mode_idx, mode_name in enumerate(self.mode_encoder.classes_):
            mask = y_test == mode_idx
            if mask.sum() > 0:
                mode_accuracy = (predicted_classes[mask] == mode_idx).mean()
                print(f"  {mode_name}: {mode_accuracy:.2%} ({mask.sum()} samples)")
        
        return loss, accuracy
    
    def save_model(self, model_path='models/learning_mode_lstm_model.keras', 
                   encoders_path='models/learning_mode_encoders.pkl'):
        """Save the trained model and encoders"""
        # Create models directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save encoders and parameters
        encoders_data = {
            'action_encoder': self.action_encoder,
            'item_type_encoder': self.item_type_encoder,
            'mode_encoder': self.mode_encoder,
            'sequence_length': self.sequence_length,
            'num_actions': self.num_actions,
            'num_item_types': self.num_item_types,
            'num_modes': self.num_modes,
            'learning_modes': self.learning_modes
        }
        
        with open(encoders_path, 'wb') as f:
            pickle.dump(encoders_data, f)
        print(f"Encoders saved to {encoders_path}")
    
    def load_trained_model(self, model_path='models/learning_mode_lstm_model.keras',
                          encoders_path='models/learning_mode_encoders.pkl'):
        """Load a trained model and encoders"""
        # Load model
        
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Load encoders
        with open(encoders_path, 'rb') as f:
            encoders_data = pickle.load(f)
        
        self.action_encoder = encoders_data['action_encoder']
        self.item_type_encoder = encoders_data['item_type_encoder']
        self.mode_encoder = encoders_data['mode_encoder']
        self.sequence_length = encoders_data['sequence_length']
        self.num_actions = encoders_data['num_actions']
        self.num_item_types = encoders_data['num_item_types']
        self.num_modes = encoders_data['num_modes']
        self.learning_modes = encoders_data['learning_modes']
        print(f"Encoders loaded from {encoders_path}")


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("LSTM Learning Mode Prediction Model Training")
    print("=" * 70)
    print("\nPredicting: sprint | adaptive_offer | archive | todays_recommendation")
    print("Based on: action_type sequences")
    
    # Initialize model
    lstm_model = LearningModePredictor(
        sequence_length=10,
        embedding_dim=32,
        lstm_units=128
    )
    
    # Build and Train Model.
    df = lstm_model.load_data()
    df = lstm_model.preprocess_data(df)
    X, y = lstm_model.create_sequences(df)
    history = lstm_model.train(X, y, epochs=30, batch_size=64)
    
    # Evaluate
    print("\n[5/5] Final evaluation...")
    # Use last 20% for testing
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    lstm_model.evaluate(X_test, y_test)
    lstm_model.save_model()
    
    # Example predictions
    print("\n" + "=" * 70)
    print("Example Predictions")
    print("=" * 70)
    
    # Test with different action patterns
    test_patterns = [
        ['enter', 'play_audio', 'respond', 'submit', 'enter', 'play_audio', 'respond', 'submit', 'quit', 'enter'],
        ['enter', 'respond', 'respond', 'respond', 'quit', 'enter', 'respond', 'quit', 'enter', 'respond'],
        ['enter', 'play_video', 'pause_video', 'play_video', 'respond', 'submit', 'enter', 'play_video', 'respond', 'submit']
    ]
    
    pattern_descriptions = [
        "Consistent progress pattern",
        "Multiple response attempts (struggling)",
        "Video learning pattern"
    ]
    
    for pattern, description in zip(test_patterns, pattern_descriptions):
        try:
            prediction = lstm_model.predict_from_action_names(pattern)
            print(f"\n{description}:")
            print(f"  Actions: {' → '.join(pattern[-5:])}")
            print(f"  Recommended Mode: {prediction['predicted_mode']}")
            print(f"  Confidence: {prediction['confidence']:.2%}")
            print(f"  All options:")
            for pred in prediction['all_predictions']:
                print(f"    - {pred['mode']:<25} {pred['probability']:.2%}")
        except ValueError:
            print(f"\n{description} {pattern}: Skipping (actions not in training data)")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
