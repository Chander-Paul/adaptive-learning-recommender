"""
Content-based recommender system that uses MultiVariateEloTopicMastery 
to recommend learning resources based on learner's knowledge state.
Uses real prerequisite data from TutorialBank dataset.
"""

import json
import os
from typing import Any, Optional

import pandas as pd
import numpy as np
from mvl_elo import MultiVariateEloTopicMastery
from learning_mode_predictor import LearningModePredictor


class ContentBasedRecommender:
    """
    Recommends learning resources based on learner's skill mastery levels
    using the MultiVariateElo topic mastery system and TutorialBank prerequisite data.
    """
    
    def __init__(self, 
                 resources_path: str = 'data/resources-v2023-clean.tsv',
                 prereq_annotations_path: str = 'data/prerequisite_annotations.csv',
                 prereq_topics_path: str = 'data/prerequisite_topics.csv',
                 default_mastery: float = 500):
        """
        Initialize the recommender with resources data and prerequisite mappings.
        
        """
        self.matcher = MultiVariateEloTopicMastery(k_factor=32, default_mastery=500, scaling_factor=400)
        self.resources = pd.read_csv(resources_path, sep='\t')
        self.default_mastery = default_mastery
        
        # Load prerequisite data
        self.prereq_topics = pd.read_csv(prereq_topics_path, quotechar='"')
        self.prereq_annotations = pd.read_csv(prereq_annotations_path, skipinitialspace=True)
        

        self.topic_prerequisites = self.load_prerequisites_data()


    
    def load_prerequisites_data(self) -> dict[str, dict[str, float]]:
        """

        """
        topic_prerequisites = {}
        
        positive_prereqs = self.prereq_annotations[
            self.prereq_annotations['prereq_relation'] == 1
        ].copy()
        
        prereq_id_to_name = dict(zip(
            self.prereq_topics['prereq_id'],
            zip(self.prereq_topics['topic_name'], self.prereq_topics['topic_url'])
        ))
        # For each resource, find its prerequisites
        for resource_id in self.resources['id'].unique():
            # Get all target topics that list this resource as having prerequisites
            resource_prereqs = positive_prereqs[
                positive_prereqs['source_topic_id'] == resource_id
            ]
            
            if len(resource_prereqs) > 0:
                prerequisites = {}
                
                # Map target_topic_id to prerequisite topic names
                for _, row in resource_prereqs.iterrows():
                    target_prereq_id = row['target_topic_id']
                    
                    if target_prereq_id in prereq_id_to_name:
                        prereq_name = prereq_id_to_name[target_prereq_id]
                        prereq_url = prereq_id_to_name[target_prereq_id]
                        
                        prerequisites[prereq_name] = self.default_mastery
                
                # Map to resource's topic name
                resource_topic = self.resources[self.resources['id'] == resource_id]['topic_name'].values
                
                if len(resource_topic) > 0:
                    topic_prerequisites[resource_topic[0]] = prerequisites
        
        return topic_prerequisites
    



    
    def recommend(self, 
                  learner_skills: dict[str, float],
                  top_n: int = 10,
                  min_readiness: float = 0.5,
                  max_readiness: float = 0.9) -> list[dict]:
        """
        Recommend learning resources based on learner's skill mastery.

        """
        recommendations = []
        
        for idx, resource in self.resources.iterrows():
            topic = resource['topic_name']
            
            # Skip if topic has no prerequisites defined
            if topic not in self.topic_prerequisites:
                continue
            
            prerequisites = self.topic_prerequisites[topic]
            
            # Calculate readiness for this topic
            comparison = self.matcher.compare_learner_to_topic(
                learner_skills, 
                prerequisites
            )
            
            readiness = comparison['overall_readiness']
            
            # Filter by readiness range (not too easy, not too hard)
            if min_readiness <= readiness <= max_readiness:
                recommendations.append({
                    'id': resource['id'],
                    'title': resource['title'],
                    'url': resource['url'],
                    'topic': topic,
                    'author': resource['author'],
                    'medium': resource['medium'],
                    'year': resource['year'],
                    'readiness_score': readiness,
                    'skill_gaps': comparison['skill_gaps'],
                    'weak_skills': comparison['weak_skills'],
                    'ready_skills': comparison['ready_skills']
                })
        
        # Sort by readiness score (optimal challenge level around 0.7)
        # Resources with readiness close to 0.7 are ideal (zone of proximal development)
        recommendations.sort(
            key=lambda x: abs(x['readiness_score'] - 0.7)
        )
        
        return recommendations[:top_n]
    

    def get_learning_path(self,
                          learner_skills: dict[str, float],
                          target_topic: str,
                          path_length: int = 5) -> list[dict]:
        """
        Generate a learning path towards a target topic by recommending tasks with higher difficulty
        """
        current_skills = learner_skills.copy()
        learning_path = []
        
        for step in range(path_length):
            # Get recommendations at current skill level
            # Gradually increase min_readiness to ensure progression
            min_readiness = 0.5 + (step * 0.05)
            max_readiness = 0.8 + (step * 0.05)
            
            recommendations = self.recommend(
                current_skills,
                top_n=5,
                min_readiness=min_readiness,
                max_readiness=max_readiness
            )
            
            if not recommendations:
                break
            
            next_resource = recommendations[0]
            learning_path.append(next_resource)
            
            # Simulate skill improvement after learning this resource
            # Assume 70% performance on learned resource
            topic_prerequisites = self.topic_prerequisites.get(
                next_resource['topic'], 
                {}
            )
            
            current_skills = self.matcher.update_learner_skills(
                current_skills,
                topic_prerequisites,
                actual_score=0.7
            )
        
        return learning_path
    
    def explain_recommendation(self, 
                              learner_skills: dict[str, float],
                              resource_id: int) -> dict:
        """
        Provide explanation for why a resource is recommended.
        
        Args:
            learner_skills: Current skill mastery levels
            resource_id: ID of the resource to explain
        
        Returns:
            Detailed explanation with skill gaps and readiness analysis
        """
        resource = self.resources[self.resources['id'] == resource_id].iloc[0]
        topic = resource['topic_name']
        prerequisites = self.topic_prerequisites.get(topic, {})
        
        comparison = self.matcher.compare_learner_to_topic(
            learner_skills,
            prerequisites
        )
        
        return {
            'resource': {
                'id': resource['id'],
                'title': resource['title'],
                'topic': topic,
                'url': resource['url']
            },
            'readiness': comparison['overall_readiness'],
            'skill_analysis': comparison['skill_matches'],
            'strengths': comparison['ready_skills'],
            'areas_to_improve': comparison['weak_skills'],
            'explanation': self.generate_explanation_text(comparison, resource)
        }
    
    def generate_explanation_text(self, comparison: dict, resource: pd.Series) -> str:
        """Generate human-readable explanation for the recommendation."""
        readiness = comparison['overall_readiness']
        ready_skills = comparison['ready_skills']
        weak_skills = comparison['weak_skills']
        
        explanation = f"This resource on '{resource['topic_name']}' is "
        
        if readiness >= 0.8:
            explanation += "well-suited to your current skill level. "
        elif readiness >= 0.6:
            explanation += "a good challenge for your current skill level. "
        else:
            explanation += "challenging but achievable with some preparation. "
        
        if ready_skills:
            skills_list = [skill for skill, _ in ready_skills]
            explanation += f"\n\nYou have strong mastery in: {', '.join(skills_list)}. "
        
        if weak_skills:
            skills_list = [skill for skill, _ in weak_skills[:3]]
            explanation += f"\n\nTo better prepare, consider strengthening: {', '.join(skills_list)}. "
        
        return explanation


class ScenarioRecommenderEngine:
    """
    Unified recommender that combines skill readiness, learning mode prediction,
    and multi-view learner profile recommendations.
    """

    def __init__(
        self,
        content_recommender: Optional[ContentBasedRecommender] = None,
        mode_predictor: Optional[LearningModePredictor] = None,
        multiview_profiles_path: str = 'visualizations/multiview_student_profiles.csv',
        multiview_recommendations_path: str = 'visualizations/multiview_recommendations.json',
        learning_paths_path: str = 'visualizations/learning_paths.json'
    ):
        self.content_recommender = content_recommender or ContentBasedRecommender()
        self.mode_predictor = mode_predictor or LearningModePredictor()
        self.mode_model_loaded = False
        self.multiview_profiles_path = multiview_profiles_path
        self.multiview_recommendations_path = multiview_recommendations_path
        self.learning_paths_path = learning_paths_path
        self.multiview_profiles_df = None
        self.multiview_recommendations = None
        self.learning_paths = None

    def recommend_for_scenario(self, scenario: dict[str, Any], top_n: int = 10) -> dict[str, Any]:
        """
        Generate recommendations based on a user scenario.

        scenario can include:
        - skills: dict[str, float]
        - learner_model: LearnerModel instance (optional)
        - recent_actions: list of [action, item_id] pairs or dicts with action/item_id
        - profile: dict of learner attributes (StudyHours, Attendance, etc.)
        - cluster_id or learning_path_cluster: output from personalized_learner_model
        - preferences: dict with optional filters (mediums, min_year, max_year)
        - target_topic: str for learning path
        """
        learner_skills = self.extract_skills(scenario)
        learning_mode = self.predict_learning_mode(scenario.get('recent_actions', []))
        readiness_range = self.readiness_range_for_mode(learning_mode.get('predicted_mode'))

        recommendations = self.content_recommender.recommend(
            learner_skills,
            top_n=top_n,
            min_readiness=readiness_range['min'],
            max_readiness=readiness_range['max']
        )

        recommendations = self.apply_preferences(recommendations, scenario.get('preferences', {}))
        recommendations = self.apply_cluster_ranking(recommendations, scenario)

        profile_info = self.infer_profile_recommendations(scenario.get('profile', {}))

        learning_path = []
        target_topic = scenario.get('target_topic')
        if target_topic:
            learning_path = self.content_recommender.get_learning_path(
                learner_skills,
                target_topic=target_topic,
                path_length=5
            )

        return {
            'learning_mode': learning_mode,
            'profile': profile_info,
            'content_recommendations': recommendations,
            'learning_path': learning_path
        }

    def extract_skills(self, scenario: dict[str, Any]) -> dict[str, float]:
        learner_model = scenario.get('learner_model')
        if learner_model is not None and hasattr(learner_model, 'get_all_skills'):
            return learner_model.get_all_skills()
        return scenario.get('skills', {})

    def predict_learning_mode(self, recent_actions: list[Any]) -> dict[str, Any]:
        if not recent_actions:
            return {'predicted_mode': 'todays_recommendation', 'confidence': 0.0, 'all_predictions': []}

        action_history = self.normalize_action_history(recent_actions)
        if not action_history:
            return {'predicted_mode': 'todays_recommendation', 'confidence': 0.0, 'all_predictions': []}

        if not self.mode_model_loaded:
            self.try_load_mode_model()

        if self.mode_model_loaded:
            try:
                return self.mode_predictor.predict_from_action_names(action_history)
            except Exception:
                return self.fallback_learning_mode(action_history)

        return self.fallback_learning_mode(action_history)

    def try_load_mode_model(self) -> None:
        try:
            self.mode_predictor.load_trained_model()
            self.mode_model_loaded = True
        except Exception:
            self.mode_model_loaded = False

    def normalize_action_history(self, recent_actions: list[Any]) -> list[list[str]]:
        normalized: list[list[str]] = []
        for entry in recent_actions:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                normalized.append([str(entry[0]), str(entry[1])])
            elif isinstance(entry, dict):
                action = entry.get('action') or entry.get('action_type')
                item_id = entry.get('item_id') or entry.get('item')
                if action and item_id:
                    normalized.append([str(action), str(item_id)])
        return normalized

    def fallback_learning_mode(self, action_history: list[list[str]]) -> dict[str, Any]:
        actions = [a[0] for a in action_history]
        item_ids = [a[1] for a in action_history]
        respond_count = sum(1 for a in actions if 'respond' in a)
        quit_count = sum(1 for a in actions if 'quit' in a)
        video_count = sum(1 for i in item_ids if 'l' in i)
        audio_count = sum(1 for a in actions if 'play_audio' in a)

        if respond_count >= 3 and quit_count >= 1:
            mode = 'adaptive_offer'
        elif video_count + audio_count >= 3:
            mode = 'archive'
        else:
            mode = 'todays_recommendation'

        return {'predicted_mode': mode, 'confidence': 0.0, 'all_predictions': []}

    def readiness_range_for_mode(self, mode: Optional[str]) -> dict[str, float]:
        mode = mode or 'todays_recommendation'
        ranges = {
            'sprint': {'min': 0.6, 'max': 0.85},
            'adaptive_offer': {'min': 0.45, 'max': 0.75},
            'archive': {'min': 0.3, 'max': 0.95},
            'todays_recommendation': {'min': 0.5, 'max': 0.85},
            'diagnosis': {'min': 0.4, 'max': 0.8}
        }
        return ranges.get(mode, ranges['todays_recommendation'])

    def apply_preferences(self, recommendations: list[dict], preferences: dict[str, Any]) -> list[dict]:
        if not recommendations:
            return recommendations

        mediums = preferences.get('mediums')
        min_year = preferences.get('min_year')
        max_year = preferences.get('max_year')

        filtered = recommendations
        if mediums:
            filtered = [r for r in filtered if str(r.get('medium', '')).lower() in {m.lower() for m in mediums}]
        if min_year is not None:
            filtered = [r for r in filtered if pd.notna(r.get('year')) and int(r['year']) >= int(min_year)]
        if max_year is not None:
            filtered = [r for r in filtered if pd.notna(r.get('year')) and int(r['year']) <= int(max_year)]

        return filtered or recommendations

    def apply_cluster_ranking(self, recommendations: list[dict], scenario: dict[str, Any]) -> list[dict]:
        if not recommendations:
            return recommendations

        cluster_id = scenario.get('learning_path_cluster') or scenario.get('cluster_id')
        if cluster_id is None:
            return recommendations

        self.load_learning_paths()
        if not self.learning_paths:
            return recommendations

        cluster_info = self.learning_paths.get(str(cluster_id))
        if not cluster_info:
            return recommendations

        target_center = self.readiness_center_for_cluster(cluster_info.get('Type', ''))
        for rec in recommendations:
            readiness = float(rec.get('readiness_score', 0.0))
            rec['cluster_score'] = 1.0 - abs(readiness - target_center)

        recommendations.sort(
            key=lambda r: (r.get('cluster_score', 0.0), 1.0 - abs(r.get('readiness_score', 0.0) - 0.7)),
            reverse=True
        )

        return recommendations

    def load_learning_paths(self) -> None:
        if self.learning_paths is not None:
            return
        if not os.path.exists(self.learning_paths_path):
            self.learning_paths = {}
            return
        with open(self.learning_paths_path, 'r', encoding='utf-8') as f:
            self.learning_paths = json.load(f)

    def readiness_center_for_cluster(self, cluster_type: str) -> float:
        cluster_type = cluster_type.upper()
        if 'HIGH ACHIEVERS' in cluster_type:
            return 0.8
        if 'STRUGGLING HARD WORKERS' in cluster_type:
            return 0.55
        if 'DISENGAGED LEARNERS' in cluster_type:
            return 0.5
        if 'CONSISTENT PERFORMERS' in cluster_type:
            return 0.7
        return 0.65

    def infer_profile_recommendations(self, profile: dict[str, Any]) -> dict[str, Any]:
        if not profile:
            return {}

        self.load_multiview_recommendations()
        if not self.multiview_recommendations:
            return {}

        composite = self.infer_composite_profile(profile)
        if not composite:
            return {}

        recs = self.multiview_recommendations.get(composite, {})
        if not recs:
            return {}

        return {
            'composite_profile': composite,
            'learning_style': recs.get('LearningStyle'),
            'performance': recs.get('Performance'),
            'affective': recs.get('Affective'),
            'recommendations': recs.get('Recommendations', [])
        }

    def load_multiview_recommendations(self) -> None:
        if self.multiview_recommendations is not None:
            return
        if not os.path.exists(self.multiview_recommendations_path):
            self.multiview_recommendations = {}
            return
        with open(self.multiview_recommendations_path, 'r', encoding='utf-8') as f:
            self.multiview_recommendations = json.load(f)

    def infer_composite_profile(self, profile: dict[str, Any]) -> Optional[str]:
        if 'Composite_Profile' in profile:
            return str(profile['Composite_Profile'])

        if all(k in profile for k in ['LearningStyle_Cluster', 'Performance_Cluster', 'Affective_Cluster']):
            return f"{profile['LearningStyle_Cluster']}_{profile['Performance_Cluster']}_{profile['Affective_Cluster']}"

        self.load_multiview_profiles()
        if self.multiview_profiles_df is None or self.multiview_profiles_df.empty:
            return None

        return self.nearest_profile_match(profile)

    def load_multiview_profiles(self) -> None:
        if self.multiview_profiles_df is not None:
            return
        if not os.path.exists(self.multiview_profiles_path):
            self.multiview_profiles_df = pd.DataFrame()
            return
        self.multiview_profiles_df = pd.read_csv(self.multiview_profiles_path)

    def nearest_profile_match(self, profile: dict[str, Any]) -> Optional[str]:
        df = self.multiview_profiles_df
        if df is None or df.empty:
            return None

        available_cols = [c for c in profile.keys() if c in df.columns]
        if not available_cols:
            return None

        numeric_cols = [c for c in available_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in available_cols if c not in numeric_cols]

        best_score = None
        best_profile = None

        for _, row in df.iterrows():
            score = 0.0
            for col in numeric_cols:
                try:
                    score += abs(float(profile[col]) - float(row[col]))
                except Exception:
                    continue
            for col in categorical_cols:
                score += 0.0 if str(profile[col]) == str(row[col]) else 1.0

            if best_score is None or score < best_score:
                best_score = score
                best_profile = row.get('Composite_Profile')

        return str(best_profile) if best_profile is not None else None



if __name__ == "__main__":
    print("=" * 80)
    print("Content-Based Learning Resource Recommender")
    print("=" * 80)
    
    # Initialize recommender
    recommender = ContentBasedRecommender()
    
    # Example learner with different skill levels
    learner = {
        'id': 'learner_001',
        'skills': {
            'general_nlp': 520,
            'machine learning': 550,
            'python': 600,
            'statistics': 540,
            'text processing': 580,
            'linguistics': 480,
            'algorithms': 500
        }
    }
    
    print(f"\nLearner Skills:")
    for skill, mastery in learner['skills'].items():
        print(f"  {skill}: {mastery}")
    
    # Get recommendations
    print("\n" + "=" * 80)
    print("Top 10 Recommended Resources:")
    print("=" * 80)
    
    recommendations = recommender.recommend(
        learner['skills'],
        top_n=10,
        min_readiness=0.5,
        max_readiness=0.85
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Topic: {rec['topic']}")
        print(f"   Readiness Score: {rec['readiness_score']:.2%}")
        print(f"   Author: {rec['author']}")
        print(f"   Medium: {rec['medium']} | Year: {rec['year']}")
        print(f"   URL: {rec['url']}")
        
        if rec['weak_skills']:
            weak_list = [f"{skill} (gap: {gap:.0f})" for skill, gap in rec['weak_skills'][:2]]
            print(f"   Skills to improve: {', '.join(weak_list)}")
    
    # Show learning path
    print("\n" + "=" * 80)
    print("Suggested Learning Path (5 resources):")
    print("=" * 80)
    
    learning_path = recommender.get_learning_path(
        learner['skills'],
        target_topic='Deep Learning for NLP',
        path_length=5
    )
    
    for i, resource in enumerate(learning_path, 1):
        print(f"\n{i}. {resource['title']}")
        print(f"   Topic: {resource['topic']}")
        print(f"   Readiness: {resource['readiness_score']:.2%}")
    
    # Detailed explanation for first recommendation
    if recommendations:
        print("\n" + "=" * 80)
        print("Detailed Explanation for Top Recommendation:")
        print("=" * 80)
        
        explanation = recommender.explain_recommendation(
            learner['skills'],
            recommendations[0]['id']
        )
        
        print(f"\nResource: {explanation['resource']['title']}")
        print(f"Topic: {explanation['resource']['topic']}")
        print(f"Overall Readiness: {explanation['readiness']:.2%}")
        print(f"\n{explanation['explanation']}")
