"""
Test Multi-View Clustering
"""
from multiview_clustering import MultiViewLearningPathClusterer
import os
import pandas as pd

# Create directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Testing Multi-View Learning Path Clustering...")
print("="*70)

option = 'predict_student'  # Change to 'predict_student' to test prediction functionality

if option == 'train_model':
# Initialize multi-view clusterer
    multiview = MultiViewLearningPathClusterer(
        data_path='data/training_files/student_performance.csv',
        use_minmax_scaler=True  # Keep all values positive
    )

    # Run the full analysis
    recommendations = multiview.run_full_multiview_analysis()
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
                                               'data/training_files/student_performance.csv',
        use_minmax_scaler=True
        )
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
    print("="*70)
    print(multiview.interpret_learning_styles(pd.DataFrame([student_features]))) 
    print(f"  Learning Style Cluster: {predicted_profile[0]} (interpretation{predicted_profile[0]})")
    print(f"  Performance Cluster: {predicted_profile[1]} (interpretation: {predicted_profile[1]})")
    print(f"  Affective Cluster: {predicted_profile[2]} (interpretation: {predicted_profile[2]})")
    print("\n✓ Student profile prediction complete!")