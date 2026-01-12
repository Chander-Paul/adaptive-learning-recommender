"""
Content-based recommender system that uses MultiVariateEloTopicMastery 
to recommend learning resources based on learner's knowledge state.
Uses real prerequisite data from TutorialBank dataset.
"""

import pandas as pd
import numpy as np


from mvl_elo import MultiVariateEloTopicMastery


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
