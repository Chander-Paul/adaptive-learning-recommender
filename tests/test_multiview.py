"""
Test Multi-View Clustering
"""
import os
from click import pause
import pandas as pd

try:
    from .multiview_clustering import MultiViewLearningPathClusterer
except ImportError:
    from multiview_clustering import MultiViewLearningPathClusterer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_files", "student_performance.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "learner")

MODEL_PATHS = [
    os.path.join(MODEL_DIR, "learning_style_clusterer.pkl"),
    os.path.join(MODEL_DIR, "performance_clusterer.pkl"),
    os.path.join(MODEL_DIR, "affective_clusterer.pkl"),
]

# Create directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Testing Multi-View Learning Path Clustering...")
print("="*70)

option = 'predict_student'  # Change to 'predict_student' to test prediction functionality

if option == 'train_model':
# Initialize multi-view clusterer
    multiview = MultiViewLearningPathClusterer(
        data_path=DATA_PATH,
        use_minmax_scaler=True  # Keep all values positive
    )

    # Run the full analysis
    cluster_profiles = multiview.run_full_multiview_analysis()
    multiview.save_model()  # Save the trained model for future use

    print("\n" + "="*70)
    print("✓ MULTI-VIEW CLUSTERING COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  visualizations/multiview_distributions.png")
    print("  visualizations/multiview_crosstabs.png")
    print("  visualizations/multiview_student_profiles.csv")
    print("  visualizations/multiview_recommendations.json")
    print("\nScatter & Silhouette plots for each view:")
    print("  visualizations/learningstyle_scatter_silhouette.png")
    print("  visualizations/performance_scatter_silhouette.png")
    print("  visualizations/affective_scatter_silhouette.png")

elif option == 'predict_student':
    multiview = MultiViewLearningPathClusterer(data_path=
                                               DATA_PATH,
        use_minmax_scaler=True
        )

    if not all(os.path.exists(path) for path in MODEL_PATHS):
        print("Saved multiview models not found at", MODEL_PATHS, ". Training them first...")
        
        pause()
        multiview.run_full_multiview_analysis()
        multiview.save_model()

    multiview.load_model()

    student_features = {
            'StudyHours': 40,
            'OnlineCourses': 15,
            'Discussions': 1,
            'Resources': 1,
            'EduTech': 1,
            'Extracurricular': 1,
            'ExamScore': 85, 'AssignmentCompletion': 90, 'FinalGrade': 4,
            'Attendance':95,'Motivation':2,'StressLevel':2
    }

    predicted_profile = multiview.predict_student_profile(student_features)
    print("\n" + "="*70)
    print("Predicted Student Profile:")
    #print(predicted_profile)
    print(predicted_profile[0])    
    print("="*70)
    #print(f"  Learning Style Cluster: {predicted_profile[0][0]} (interpretation: {predicted_profile[0][0])})")
    #print(f"  Performance Cluster: {predicted_profile[0][1]} (interpretation: {predicted_profile[0][1]})")
    #print(f"  Affective Cluster: {predicted_profile[0][2]} (interpretation:{predicted_profile[0][2].})")
    print("\n✓ Student profile prediction complete!")