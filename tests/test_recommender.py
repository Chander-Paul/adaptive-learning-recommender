import os
import pathlib
import sys



PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep tests independent of shell CWD so relative data paths in app code resolve.
os.chdir(PROJECT_ROOT)

import unittest
import recommender

from models.learner.multiview_clustering import MultiViewLearningPathClusterer
from models.learner import learner_model as learner
from adaptive_recommender import AdaptiveRecommender
import pandas as pd
import helper_functions as helper


## load config file data
cfg, root = helper.load_config()
embedding_dir = root / cfg["embeddings"]["embedding_path"]   # "embeddings/"
model_dir = root / cfg["models"]["path"]                     # "models/"
data_dir = root / cfg["data"]["path"]                       # "data/"
topic_file = data_dir / cfg["data"]["topics_file"]          # "topics_mappings.csv"
print(f"Embedding Directory: {embedding_dir}")

class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.recommender = recommender.ResourceRecommender()
        self.student = learner.LearnerModel(learner_id="1",name = "Jane Doe")
        self.topics = pd.read_csv(topic_file, skiprows=1).head(10)

    def test_loginteraction(self):
        interaction = learner.ContentInteraction(   
            content_id="10",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        self.student.log_content_interaction(interaction)
        interaction = learner.ContentInteraction(
            content_id="11",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        self.student.log_content_interaction(interaction)
        interaction = learner.ContentInteraction(
            content_id="12",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        self.student.log_content_interaction(interaction)
        print(self.student.content_interactions)
        
    def test_historicalrecomendation(self):
        """Test a simple case"""
        self.setup()
        self.student.log_content_interaction(interaction)
        interaction = learner.ContentInteraction(
            content_id="10",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        
        self.student.log_content_interaction(interaction)
        interaction = learner.ContentInteraction(
            content_id="12",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        self.student.log_content_interaction(interaction)

    def test_recommendation_with_user_embeddings(self):
        """ With user embeddings generated from recent content."""
        self.setUp()
        interaction = learner.ContentInteraction(
            content_id="11",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        
        self.student.log_content_interaction(interaction)
        interaction = learner.ContentInteraction(
            content_id="11",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        self.student.log_content_interaction(interaction)       
        self.recommender.load_embeddings()
        interactions = self.student.content_interactions
        for interaction in interactions:
            print(interaction.__dict__)
        
       ## print("Resource Embededers", list(self.recommender.resource_embeddings.items())[:10])
        self.student.learner_embeddings = self.student.build_learner_embedding(self.recommender.resource_embeddings)
        self.recommender.ENABLE_SCORING = False
        self.recommender.ENABLE_ENGAGEMENT_RERANKING = False
        recommendation = self.recommender.recommend(self.student, self.topics)
        print("Authorship Reommendations")
        for rec in recommendation:
            print(rec["score"], rec["title"], rec["url"])
    
    def test_basic_recommendationfull(self):
        """Test a simple case"""
        
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


        
        self.assertTrue(True)
        print("Testing Score Resource")
        candidates = self.recommender.apply_preferences(self.recommender.resources.copy(), sample_profile)
        print(candidates)

    
   
    
        self.assertTrue(True)
        print("Testing Score Resource")
        candidates = self.recommender.apply_preferences(self.recommender.resources.copy(), sample_profile)
        print(candidates)


class TestLearnerModel(unittest.TestCase):
    def setUp(self):
        self.learner = learner.LearnerModel(learner_id="1", name="Jane Doe")
        self.clusterer = MultiViewLearningPathClusterer()
        self.clusterer.load_model(cfg["models"]["learner_model_path"])  # Load the trained multiview clusterer

    def test_learnercluster_classification(self):
        """Test if the learner is classified into the correct cluster based on their profile."""
        sample_study_habits_profile = {
            'StudyHours': 40,
            'OnlineCourses': 15,
            'Discussions': 1,
            'Resources': 1,
            'EduTech': 1,
            'Extracurricular': 1,
            'ExamScore': 85, 'AssignmentCompletion': 90, 
            'Attendance':95,'Motivation':2,'StressLevel':2
        }
        self.learner.study_habits = learner.StudyHabits(
            weekly_study_hours= sample_study_habits_profile["StudyHours"],
            online_courses=sample_study_habits_profile["OnlineCourses"],
            discussions_participated=sample_study_habits_profile["Discussions"],
            resources_used=sample_study_habits_profile["Resources"],
            educational_technology_used=sample_study_habits_profile["EduTech"],
        )

        predicted_profile = self.learner.predict_learner_profile(sample_study_habits_profile, self.clusterer)
        print("Predicted Learner Profile:", predicted_profile)
        self.assertIsNotNone(predicted_profile)

class TestAdaptiveModePredictor(unittest.TestCase):
    def setUp(self):
        from models.learner.learning_mode_predictor import LearningModePredictor
        self.predictor = LearningModePredictor()
        self.predictor.load_trained_model()

    def test_predict_from_action_patterns(self):
        """Test if the predictor can generate a recommendation from action patterns."""
        test_pattern = [['enter','q'], ['play_audio','l'], ['respond','q'], ['submit','q'], ['enter','l'], ['play_video','l'], ['pause_video','l'], ['play_video','l']]
        prediction = self.predictor.predict_from_action_names(test_pattern)
        print("Predicted Learning Mode:", prediction)
        self.assertIn('predicted_mode', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('all_predictions', prediction)

    def test_adaptive_recommender_integration(self):
        """Test if the AdaptiveRecommender can generate recommendations based on predicted learning modes."""
        
        adaptive_recommender = AdaptiveRecommender()    
        self.student = learner.LearnerModel(learner_id="1", name="Jane Doe")

        learner_profile = {
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
        interaction = learner.ContentInteraction(
            content_id="11",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        self.student.log_content_interaction(interaction)
        interaction = learner.ContentInteraction(
            content_id="10",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        
        self.student.log_content_interaction(interaction)
        interaction = learner.ContentInteraction(
            content_id="12",
            topic_id="1",
            timestamp="2024-01-01T10:00:00Z",
            duration_seconds=300,
            feedback=4,
            completed = False
        )
        self.student.log_content_interaction(interaction)



        
        recent_actions = [['enter','q'], ['play_audio','l'], ['respond','q'], ['submit','q'], ['enter','l'], ['play_video','l'], ['pause_video','l'], ['play_video','l']]
        
        recommendations, predicted_mode = adaptive_recommender.recommend(
            profile=self.student,
            recent_actions=recent_actions,
            top_n=5,
                triggered_resource=True,
                trigger_content_id="10"
        )
        print("Adaptive Recommendations:"),
        for recommendation in recommendations:
            print(f"- {recommendation['title']} (Score: {recommendation['score']})")
        print("Predicted Learning Mode:", predicted_mode)
        print("Number of Recommendations:", len(recommendations))

        self.assertIsInstance(recommendations, list)
        self.assertIsInstance(predicted_mode, str)


if __name__ == '__main__':
    unittest.main()
