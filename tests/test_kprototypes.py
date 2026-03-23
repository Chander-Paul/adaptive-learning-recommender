from models.learner.kprototypes_learner_model import KPrototypesLearningPathClusterer


if __name__ == "__main__":

    """Main execution function."""
    # Create directories if they don't exist
    import os
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize and run the clustering model
    clusterer = KPrototypesLearningPathClusterer(
        data_path='data/training_files/student_performance.csv'
    )
    learning_paths = clusterer.run_full_analysis()
    
    # Example: Predict cluster for a new student
    print("\n" + "="*60)
    print("EXAMPLE: PREDICT LEARNING PATH FOR NEW STUDENT")
    print("="*60)
    
    new_student = {
        'StudyHours': 25,
        'Attendance': 90,
        'Age': 22,
        'AssignmentCompletion': 85,
        'ExamScore': 75,
        'OnlineCourses': 10,
        'Resources': 1,
        'Extracurricular': 1,
        'Motivation': 2,
        'Internet': 1,
        'Gender': 1,
        'LearningStyle': 2,
        'EduTech': 1,
        'StressLevel': 1,
        'Discussions': 1
    }
    
    predicted_cluster = clusterer.predict_cluster(new_student)
    print(f"\nNew student predicted to be in Cluster {predicted_cluster}")
    print(f"Learning Path: {learning_paths[predicted_cluster]['Type']}")
    print(f"Description: {learning_paths[predicted_cluster]['Description']}")
    print("\nRecommendations:")
    for rec in learning_paths[predicted_cluster]['Recommendations']:
        print(f"  • {rec}")

