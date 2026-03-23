"""
Evaluation for main recommender

Tracking Relevancy and engagement of receommednations
Basic precision and recall done manually, 
Done with max number of returned values to allow for precision @k and recall @k performed after original checks.,

Sepreate evaluations will be done for each individal model
"""

from  recommender import ResourceRecommender


if __name__ == "__main__":
    query = "Machine Learning"
    recommender = ResourceRecommender()

    sample_profile = {
        "StudyHours": 12,
        "OnlineCourses": 6,
        "Discussions": 1,
        "Resources": 1,
        "EduTech": 1,
        "Extracurricular": 0,
        "ExamScore": 78,
        "AssignmentCompletion": 70,
        "Attendance": 82,
        "Motivation": 1,
        "StressLevel": 2,
        "preferences": {"mediums": ["Tutorials", "Resources"], "min_date": 2008},
    }
    
    
    top = recommender.recommend(sample_profile, top_n=5, query=query)
    print("Recommendations")

    for i, rec in enumerate(top, 1):
        print(f"{i}. {rec['title']} ({rec['medium']}, {rec['url']}) -> score={rec['score']}")
    
    print("\nEvaluation Metrics:")