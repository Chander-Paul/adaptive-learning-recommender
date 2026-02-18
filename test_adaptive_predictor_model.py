
from json import load
from learning_mode_predictor import LearningModePredictor


lstm_model = LearningModePredictor()
lstm_model.load_trained_model()

# Example predictions
print("\n" + "=" * 70)
print("Example Predictions")
print("=" * 70)

# Test with different action patterns


test_patterns = [
    [['enter','q213'], ['play_audio','l1234'], ['respond','q1234'], ['submit','q1234'], ['enter','l5678'], ['play_video','l5678'], ['pause_video','l5678'], ['play_video','l5678']],
    [['enter','q213'], ['play_audio','l1234'], ['respond','q1234'], ['submit','q1234'], ['respond','q1234'], ['submit','q1234'], ['respond','q1234'], ['submit','q1234']],  
    [['enter','q213'], ['play_video','l5678'], ['pause_video','l5678'], ['play_video','l5678'], ['pause_video','l5678'], ['play_video','l5678']]

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
        print(f"  Actions: {' → '.join([p[0] for p in pattern[-5:]])}")
        print(f"  Recommended Mode: {prediction['predicted_mode']}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"  All options:")
        for pred in prediction['all_predictions']:
            print(f"    - {pred['mode']:<25} {pred['probability']:.2%}")
    except ValueError:
        print(f"\nError {description} {pattern}: Skipping (actions not in training data)")

print("\n" + "=" * 70)
print("Testing Complete!")
print("=" * 70)