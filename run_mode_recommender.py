"""Script that predicts learning mode and calls the resource recommender."""

from __future__ import annotations

import argparse
import json
from typing import Any

from learning_mode_predictor import LearningModePredictor
from learner_type_recommender import ResourceRecommender
import recommender


def _load_json_arg(value: str, label: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {exc}") from exc

def recommend_from_interaction(recent_actions: str, profile_json: str, top_n: int = 5, query: str = None) -> list[dict[str, Any]]:
    
    profile = _load_json_arg(profile_json, "profile-json") if profile_json else {}

    predictor = LearningModePredictor()
    predictor.load_trained_model()
    prediction = predictor.predict_from_action_names(recent_actions)
    if prediction.get("predicted_mode") == "adaptive_offer":
        recommender = ResourceRecommender()
        if prediction.get("resource_type", 0) == 'question':
            recommendations = recommender.recommend(profile, top_n=top_n, query=query, question=1)
        else:
            recommendations = recommender.recommend(profile, top_n=top_n, query=query, question=0)
    return recommendations
    

if __name__ == "__main__":
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict learning mode and recommend resources.")
    parser.add_argument("--actions-json", type=str, help="JSON list of [action, item_id] pairs")
    parser.add_argument("--profile-json", type=str, help="JSON profile dict for the recommender")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--query", type=str, default=None, help="Optional query to seed content similarity")
    args = parser.parse_args()

    if args.actions_json:
        recent_actions = _load_json_arg(args.actions_json, "actions-json")
    else:
        recent_actions = [
            ["enter", "l1"],
            ["play_audio", "l2"],
            ["respond", "q5"],
            ["submit", "q5"],
            ["quit", "l2"],
        ]

    profile = _load_json_arg(args.profile_json, "profile-json") if args.profile_json else {}

    predictor = LearningModePredictor()
    predictor.load_trained_model()
    prediction = predictor.predict_from_action_names(recent_actions)
    if prediction.get("predicted_mode") == "adaptive_offer":
        recommender = ResourceRecommender()
        if prediction.get("resource_type", 0) == 'question':
            recommendations = recommender.recommend(profile, top_n=args.top_n, query=args.query, question=1)
        else:
            recommendations = recommender.recommend(profile, top_n=args.top_n, query=args.query, question=0)

    

if __name__ == "__main__":
    main()
