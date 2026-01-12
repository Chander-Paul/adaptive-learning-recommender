""" Implementation fo a multivariate elo-based model to determine learner readiness
    for topics based on skills previously learned. 

    Will be extended to include additional factors such as time spent learning a skill,
    recency, study freqyency and assessment scores. 

    This algorittm is absed on the elo rating systemn ofr Arpad elo.

"""

import numpy as np


class MultiVariateEloTopicMastery:

    def __init__(self, k_factor: float = 32, default_mastery: float = 500, scaling_factor: float = 400):


        self.k_factor = k_factor
        self.default_mastery = default_mastery
        self.scaling_factor = scaling_factor
    
    
    def calculate_skill_match(self, learner_mastery: float, required_mastery: float) -> float:
        """ Calculate the probability that the learner meets the required mastery for a skill in a topic 
        using an Elo-based formula.
        """

        return 1.0 / (1.0 + 10.0 ** ((required_mastery - learner_mastery) / self.scaling_factor))
    
    def compare_learner_to_topic(self, learner_skills: dict[str, float], topic_prerequisites: dict[str, float]) -> dict:
        """ Check if the Learner meets the prerequisite requirement for a topic based on skill elo."""
        
        if not topic_prerequisites:
            return {
                'overall_readiness': 1.0,
                'skill_matches': {},
                'skill_gaps': {},
                'ready_skills': [],
                'weak_skills': []
            }
        
        match_scores = []
        weights = []
        skill_matches = {}
        skill_gaps = {}
        ready_skills = []
        weak_skills = []
        
        for skill, required_mastery in topic_prerequisites.items():
            learner_mastery = learner_skills.get(skill, self.default_mastery)
            match_score = self.calculate_skill_match(learner_mastery, required_mastery)

           
            
            match_scores.append(match_score)
            weights.append(required_mastery)
            
            skill_matches[skill] = {
                'match_probability': match_score,
                'learner_mastery': learner_mastery,
                'required_mastery': required_mastery
            }
            
            gap = required_mastery - learner_mastery
            
            if gap <= 0:
                ready_skills.append(skill)
            else:
                weak_skills.append((skill, gap))
            skill_gaps[skill] = gap

        
        # Get readiness based on weighted average of skill match scores
        match_scores = np.array(match_scores)
        weights = np.array(weights)
        weights = weights / weights.sum()
        overall_readiness = float(np.dot(match_scores, weights))
        
        return {
            'overall_readiness': overall_readiness,
            'skill_matches': skill_matches,
            'skill_gaps': skill_gaps,
            'ready_skills': ready_skills,
            'weak_skills': weak_skills
        }
    
    def update_learner_skills(self, 
                             learner_skills: dict[str, float],
                             topic_prerequisites: dict[str, float],
                             actual_score: float) -> dict[str, float]:
        
        """ Update the learner's skill mastery levels based on their performance in a topic.
        """
        updated_skills = learner_skills.copy()
        
        for skill, required_mastery in topic_prerequisites.items():
            current_mastery = learner_skills.get(skill, self.default_mastery)
            expected_score = self.calculate_skill_match(current_mastery, required_mastery)
            
            # ELO update formula
            mastery_change = self.k_factor * (actual_score - expected_score)
            new_mastery = current_mastery + mastery_change
            
            updated_skills[skill] = new_mastery
        
        return updated_skills


# Test
if __name__ == "__main__":
    """ Simple test of the MultiVariateEloTopicMastery class """
    matcher = MultiVariateEloTopicMastery(k_factor=32, default_mastery=500, scaling_factor=400)
    
    # Define one learner skills and topic prerequisites. 
    
    learner={'id': "learner_001",
        'skills': {
            'python': 600,
            'statistics': 550,
            'machine learning': 450
        }
    }
    
    topic={'name': "Introduction to Machine Learning",
           'prerequisites': {
               'python': 500,
               'statistics': 600,
               'machine learning': 500
               }
            }
    
    print(f"{learner['id']} Skills")
    for skill, mastery in learner['skills'].items():
        print(f"  {skill}: {mastery}")
    print()
    
    print(f"{topic['name']  } Prerequisites")
    for skill, required in topic['prerequisites'].items():
        print(f"  {skill}: {required}")
    print()
    

    print("Comparison Results")
    comparison = matcher.compare_learner_to_topic(learner['skills'], topic['prerequisites'])
    
    print(f"Overall Readiness: {comparison['overall_readiness']:.2%}\n")
    
    print("Skill Mastery Readiness:")
    for skill, match_info in comparison['skill_matches'].items():
        print(f"  {skill}:")
        print(f"   Learner Mastery: {match_info['learner_mastery']:.0f}")
        print(f"   Required Mastery: {match_info['required_mastery']:.0f}")
        print(f"   Match Probability: {match_info['match_probability']:.2%}")
        print(f"   Gap: {comparison['skill_gaps'][skill]:.0f}")
    print()
    
    print(f"Ready Skills: {comparison['ready_skills']}")
    print(f"Weak Skills (skill, gap): {comparison['weak_skills']}")
    print()
    
    # Simulate topic completion and update skills
    print("=== After topic Completion ===")
    print("Learner completes topic with 75% performance")
    updated_skills = matcher.update_learner_skills(
        learner_skills=learner['skills'], 
        topic_prerequisites=topic['prerequisites'], 
        actual_score=0.75
    )
    
    print("\nUpdated Skills:")
    for skill, mastery in updated_skills.items():
        old_mastery = learner['skills'].get(skill, matcher.default_mastery)
        change = mastery - old_mastery
        print(f"  {skill}: {old_mastery:.0f} → {mastery:.0f} ({change:+.0f})")

    
    # Compare again with updated skills
    print("=== Re-evaluation with Updated Skills ===")
    new_comparison = matcher.compare_learner_to_topic(updated_skills, topic['prerequisites'])
    print(f"New Overall Readiness: {new_comparison['overall_readiness']:.2%}")
    print(f"Readiness Change: {new_comparison['overall_readiness'] - comparison['overall_readiness']:+.2%}")
