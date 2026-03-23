"""
Quick Reference: Using the Learning Mode Predictor with Item Types
"""




# Basic Run


# Load trained model
from learning_mode_predictor import LearningModePredictor
import pandas as pd

model = LearningModePredictor()
model.load_trained_model()

# Predict with actions and item types
actions = ['enter', 'play_video', 'respond', 'submit', 'quit']
item_types = ['lecture', 'lecture', 'question', 'question', 'lecture']

prediction = model.predict_from_action_names(actions, item_types)
print(f"Recommended: {prediction['predicted_mode']} ({prediction['confidence']:.1%})")



# Item Extractio from data


# Method 1: From item_id in CSV
df = pd.read_csv('data/training_files/KT4Subset/u20.csv')

# Extract item type from first character
df['item_type'] = df['item_id'].str[0].map({
    'b': 'block',
    'e': 'explanation',
    'q': 'question', 
    'l': 'lecture'
})

# Get last 10 interactions
recent = df.tail(10)
actions = recent['action_type'].tolist()
item_types = recent['item_type'].tolist()

prediction = model.predict_from_action_names(actions, item_types)



# Example user input pattern and expected outcomes


# Pattern 1: Lecture Video watching
actions_1 = ['enter', 'play_video', 'pause_video', 'play_video', 'quit',
             'enter', 'play_video', 'pause_video', 'quit', 'enter']
items_1 = ['lecture'] * 10

pred_1 = model.predict_from_action_names(actions_1, items_1)
print(f"Pattern 1: {pred_1['predicted_mode']}")  # Likely: archive


# Pattern 2: Struggling with questions = offer Adaptive offer
actions_2 = ['enter', 'respond', 'respond', 'erase_choice', 'respond',
             'quit', 'enter', 'respond', 'respond', 'quit']
items_2 = ['question'] * 10

pred_2 = model.predict_from_action_names(actions_2, items_2)
print(f"Pattern 2: {pred_2['predicted_mode']}")  # Likely: todays_recommendation


# Pattern 3: Steady progress -> Offer Sprint
actions_3 = ['enter', 'play_audio', 'respond', 'submit', 'enter',
             'play_audio', 'respond', 'submit', 'enter', 'respond']
items_3 = ['block', 'block', 'question', 'question', 'block',
           'block', 'question', 'question', 'block', 'question']

pred_3 = model.predict_from_action_names(actions_3, items_3)
print(f"Pattern 3: {pred_3['predicted_mode']}")  # Likely: sprint





# Available course items

(b) block     - Learning blocks/units
(q) question     - Questions to answer
(e) explanation - Explanatory content
(l) lecture     - Lecture videos
(u) unknown     - When item_id not provided



Learning Modes:
sprint               - Focused short courses
adaptive_offer       - Help after struggles
archive              - Browsable lectures
todays_recommendation - Daily personalized content