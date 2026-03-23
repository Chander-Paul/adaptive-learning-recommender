"""
Uses predictions from learning_mode_predictor module to call the recommender function.
Based on the prediction it will limit the recommender to types of resources that are most effective
for the learner's current situation.

"""
from typing import Any

from models.learner.learning_mode_predictor import LearningModePredictor
from recommender import ResourceRecommender
class AdaptiveRecommender:
    def __init__(self):
        self.predictor = LearningModePredictor()
        self.predictor.load_trained_model()
        self.recommender = ResourceRecommender()

    def recommend(self, profile=None, recent_actions: list[tuple[str, str]] = None,
                   top_n: int = 10,
                    triggered_resource: bool = False,
                    trigger_content_id: str | None = None) -> list[dict[str, Any]]:
        prediction = self.predictor.predict_from_action_names(recent_actions)
        predicted_mode = prediction.get('predicted_mode')
        
        # Define filters based on predicted learning mode
        filters = []
        if predicted_mode == 'sprint':
            filters.append({"key": "type", "value": "resource"})
        elif predicted_mode == 'assignment':
            filters.append({"key": "type", "value": "assignments"}) ##exercise is a better term than assignments
        elif predicted_mode == 'instructional_material':
            filters.append({"key": "type", "value": "resources"})
        elif predicted_mode == 'adaptive_offer':
            filters.append({"key": "type", "value": "resources"})
        elif predicted_mode == 'todays_recommendation':
            filters.append({"key": "type", "value": "resources"})

        # Call the recommender with the appropriate filters
        if triggered_resource and trigger_content_id:
            recommendations = self.recommender.recommend(profile, top_n=top_n, filter=filters, similar_to_content_id=trigger_content_id)
        else:
            recommendations = self.recommender.recommend(profile, top_n=top_n, filter=filters)
        
        return recommendations, predicted_mode
