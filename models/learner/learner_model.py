"""Class for managing a learner object
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json
from .multiview_clustering import MultiViewLearningPathClusterer
from .mvl_elo import MultiVariateEloTopicMastery
import numpy as np


@dataclass
class SkillMastery:
    """Represents a learner's mastery of a skill."""
    
    skill_name: str
    mastery_score: float
    last_updated: datetime
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, new_score: float, context: Optional[Dict[str, Any]] = None):
        """Update the mastery score and track history."""
        self.history.append({
            'timestamp': self.last_updated.isoformat(),
            'old_score': self.mastery_score,
            'new_score': new_score,
            'change': new_score - self.mastery_score,
            'context': context or {}
        })
        self.mastery_score = new_score
        self.last_updated = datetime.now()
    
    



@dataclass
class ContentInteraction:
    """
        Represents a learner's interaction with learning content
    """
    
    content_id: str
    topic_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: int = 0
    completed: bool = False
    performance_score: Optional[float] = None # Only for assignment type interactions
    engagement_score: Optional[float] = None
    skills_practiced: List[str] = field(default_factory=list)
    feedback: int = None
    


@dataclass
class StudyHabits:
    """Represents a learner's study habits and preferences."""
    
    weekly_study_hours: Optional[float] = None
    online_courses: Optional[int] = None
    discussions_participated: Optional[int] = None
    resources_used: Optional[int] = None
    educational_technology_used: Optional[int] = None
    average_session_length_minutes: Optional[float] = None
    preferred_content_types: List[str] = field(default_factory=list)  # e.g., audio, visual, kinesthetic
    study_environment: Optional[str] = None  # dsicussion, online, selfstudy


    preferred_study_times: List[str] = field(default_factory=list)  # e.g., ["morning", "evening"]
    
    



class LearnerModel:
    """
        Model for tracking a learner's knowledge state and skill mastery.
    """
    
    def __init__(
        self,
        learner_id: str,
        name: str = "",
        elo_matcher: Optional[MultiVariateEloTopicMastery] = None,
        default_mastery: float = 500
    ):
        """Initialize a learner model.
        
        Args:
            learner_id: Unique identifier for the learner
            name: Learner's name
            elo_matcher: MultiVariateEloTopicMastery instance for skill calculations
            default_mastery: Default mastery score for new skills
        """
        self.learner_id = learner_id
        self.name = name
        self.registration_date = datetime.now()
        self.default_mastery = default_mastery

        


        
        # Initialize Elo matcher
        # Liming to one elo matcher per learner before extending to other courses.
        self.elo_matcher = elo_matcher or MultiVariateEloTopicMastery(
            k_factor=32,
            default_mastery=default_mastery,
            scaling_factor=400
        )
        self.skills: Dict[str, SkillMastery] = {}       # Skill tracking
        self.content_interactions: List[ContentInteraction] = []        # Learning history
        self.study_habits: StudyHabits = None
        # Statistics
        self.total_learning_time = 0  # in seconds
        self.topics_completed = 0
        self.total_interactions = 0
        self.learner_embeddings = None
        self.learner_profile = None
        self.learning_style_cluster = None
        self.performance_cluster = None
        self.affective_cluster = None
    
    
    def log_content_interaction(self, interaction: ContentInteraction):
        """Add a content interaction to the learner's history."""
        interaction.engagement_score = self.content_engagement_calculation(interaction)
        self.content_interactions.append(interaction)
        self.total_learning_time += interaction.duration_seconds
        self.total_interactions += 1
        if interaction.completed:
            self.topics_completed += 1  

        

    
    #### Skill and Topic Readiness management methods ####
    def get_skill_mastery(self, skill_name: str) -> float:
        """Get the current mastery score for a skill.
        
        Returns default_mastery if skill hasn't been encountered yet.
        """
        if skill_name in self.skills:
            return self.skills[skill_name].mastery_score
        return self.default_mastery
    
    def get_all_skills(self) -> Dict[str, float]:
        """Get all skills and their current mastery scores."""
        return {name: skill.mastery_score for name, skill in self.skills.items()}
    
    def add_skill(self, skill_name: str, initial_mastery: Optional[float] = None):
        """Add a new skill to the learner's profile."""
        if skill_name not in self.skills:
            mastery = initial_mastery if initial_mastery is not None else self.default_mastery
            self.skills[skill_name] = SkillMastery(
                skill_name=skill_name,
                mastery_score=mastery,
                last_updated=datetime.now()
            )
    
    def update_skill(self, skill_name: str, new_mastery: float, context: Optional[Dict] = None):
        """Update a skill's mastery score."""
        if skill_name not in self.skills:
            self.add_skill(skill_name)
        
        self.skills[skill_name].update(new_mastery, context)
    
    def check_readiness_for_topic(
        self,
        topic_name: str,
        topic_prerequisites: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check if learner is ready for a topic based on prerequisite skills.
        
        Args:
            topic_name: Name of the topic
            topic_prerequisites: Dictionary of {skill: required_mastery}
        
        Returns:
            Dictionary with readiness analysis including overall_readiness,
            skill_matches, skill_gaps, ready_skills, and weak_skills
        """
        learner_skills = self.get_all_skills()
        
        # Ensure all prerequisite skills exist in learner profile
        for skill in topic_prerequisites:
            if skill not in learner_skills:
                learner_skills[skill] = self.default_mastery
        
        comparison = self.elo_matcher.compare_learner_to_topic(
            learner_skills,
            topic_prerequisites
        )
        
        comparison['topic_name'] = topic_name
        comparison['timestamp'] = datetime.now().isoformat()
        
        return comparison
    
    def record_topic_completion(self, content_id: str, topic_name: str, topic_prerequisites: Dict[str, float],
        performance_score: float, duration_seconds: int, engagement_score: Optional[float] = None, completed: bool = True
    ) -> Dict[str, float]:
        """Record a topic completion and update skill mastery levels.
        """
        # Record the interaction
        interaction = ContentInteraction(
            content_id=content_id,
            topic_name=topic_name,
            timestamp=datetime.now(),
            duration_seconds=duration_seconds,
            completed=completed,
            performance_score=performance_score,
            engagement_score=engagement_score,
            skills_practiced=list(topic_prerequisites.keys())
        )
        self.content_interactions.append(interaction)
        
        # Update statistics
        self.total_learning_time += duration_seconds
        self.total_interactions += 1
        if completed:
            self.topics_completed += 1
        
        # Update skill mastery using Elo system
        current_skills = self.get_all_skills()
        updated_skills = self.elo_matcher.update_learner_skills(
            learner_skills=current_skills,
            topic_prerequisites=topic_prerequisites,
            actual_score=performance_score
        )
        
        # Update each skill in the learner model
        context = {
            'content_id': content_id,
            'topic_name': topic_name,
            'performance_score': performance_score,
            'completed': completed
        }
        
        for skill_name, new_mastery in updated_skills.items():
            self.update_skill(skill_name, new_mastery, context)
        
        return updated_skills
    
    def get_skill_progression(self, skill_name: str) -> List[Dict[str, Any]]:
        """Get the progression history for a specific skill."""
        if skill_name not in self.skills:
            return []
        return self.skills[skill_name].history
    
    def get_weak_skills(self, threshold: float = 500) -> List[tuple[str, float]]:
        """Get skills below a certain mastery threshold.
        
        Args:
            threshold: Mastery score threshold
        
        Returns:
            List of (skill_name, mastery_score) tuples
        """
        return [
            (name, skill.mastery_score)
            for name, skill in self.skills.items()
            if skill.mastery_score < threshold
        ]
    
    def get_strong_skills(self, threshold: float = 600) -> List[tuple[str, float]]:
        """Get skills above a certain mastery threshold.
        
        Args:
            threshold: Mastery score threshold
        
        Returns:
            List of (skill_name, mastery_score) tuples
        """
        return [
            (name, skill.mastery_score)
            for name, skill in self.skills.items()
            if skill.mastery_score >= threshold
        ]
    

    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get overall learning statistics."""
        avg_performance = None
        if self.content_interactions:
            completed_with_scores = [
                i.performance_score for i in self.content_interactions
                if i.performance_score is not None
            ]
            if completed_with_scores:
                avg_performance = sum(completed_with_scores) / len(completed_with_scores)
        
        return {
            'learner_id': self.learner_id,
            'total_skills': len(self.skills),
            'total_interactions': self.total_interactions,
            'topics_completed': self.topics_completed,
            'total_learning_time_hours': self.total_learning_time / 3600,
            'average_performance': avg_performance,
            'registration_date': self.registration_date.isoformat()
        }
    


    

    ### User Content Engagement Methods ###
    def content_engagement_calculation(self, interaction: ContentInteraction, weights = [0.4, 0.3, 0.3]) -> Optional[float]:
        """Calculate and store engagement score for interaction with content.
        Calculation is based on duration, feedback, and completion status, with the following weights:

        Engagement metrics are stored later to be piped into a an untrained model. For collobaritive searching. 
        """
        if interaction is None:
            return None

        duration_seconds = max(interaction.duration_seconds or 0, 0)
        feedback_value = interaction.feedback
        completed_value = 1.0 if interaction.completed else 0.0

        duration_score = min(duration_seconds / 3600.0, 1.0)

        feedback_score = 0.0
        if feedback_value is not None:
            bounded_feedback = min(max(float(feedback_value), 1.0), 5.0)
            feedback_score = (bounded_feedback - 1.0) / 4.0

        engagement_score = (
            weights[0] * duration_score
            + weights[1] * feedback_score
            + weights[2] * completed_value
        )

        interaction.engagement_score = round(engagement_score, 4)
        return interaction.engagement_score
    
    
    def get_recent_interactions(self, limit: int = 10) -> List[ContentInteraction]:
        """Get the most recent content interactions."""
        return sorted(
            self.content_interactions,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]

   
    
    def get_engaged_content(self, threshold: float = 0.5) -> List[ContentInteraction]:
        """Get content interactions with engagement score above a certain threshold."""
        engaged_content = []
        for interaction in self.content_interactions:
            if interaction.engagement_score is None:
                self.content_engagement_calculation(interaction)
            if interaction.engagement_score is not None and interaction.engagement_score >= threshold:
                engaged_content.append(interaction)
        return engaged_content
    
    
    

    def build_learner_embedding( self, content_embeddings: Dict[str, np.ndarray],
        strategy: str = "engagement_weighted", recency_decay: float = 0.95,
        limit: int = 100
    ) -> Optional[np.ndarray]:
        """Build a learner embedding by aggregating historical content embeddings.

        Args:
            content_embeddings: Dict mapping content_id -> embedding vector (float32)
            strategy: One of 'mean', 'engagement_weighted', 'recency_weighted', 'combined'
            recency_decay: Decay factor per interaction (most recent = 1.0)

        Returns:
            Aggregated learner embedding as np.ndarray, or None if no matches found
        """
        interactions = self.get_recent_interactions(limit=limit)

        vectors = []
        weights = []

        for i, interaction in enumerate(interactions):
            embedding = content_embeddings.get(int(interaction.content_id))
            if embedding is None:
                print(f"Warning: No embedding found for content_id {interaction.content_id}")
                ## Add code for adding missing resources to embedding training list ##
                continue

            vectors.append(np.array(embedding, dtype=np.float32))

            if strategy == "mean":
                weight = 1.0

            elif strategy == "engagement_weighted":
                weight = interaction.engagement_score or 0.1

            elif strategy == "recency_weighted":
                # More recent interactions have higher weight
                recency_index = i - len(interactions)  # negative, closer to 0 = more recent
                weight = recency_decay ** abs(recency_index)

            elif strategy == "combined":
                engagement = interaction.engagement_score or 0.1
                recency_index = i - len(interactions)
                recency = recency_decay ** abs(recency_index)
                weight = engagement * recency
            else:
                weight = 1.0

            weights.append(weight)

        if not vectors:
            return None

        vectors = np.stack(vectors)          # shape: (n, embedding_dim)
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()             # normalise

        learner_embedding = np.average(vectors, axis=0, weights=weights)

        # L2 normalise for cosine similarity compatibility
        norm = np.linalg.norm(learner_embedding)
        if norm > 0:
            learner_embedding /= norm

        self.learner_embeddings = {
            "vector": learner_embedding,
            "dim": learner_embedding.shape[0],
            "strategy": strategy,
            "updated_at": datetime.now().isoformat(),
            "interaction_count": len(vectors)
        }

        return learner_embedding
    
    def predict_learner_profile(self, student_features: Dict[str, Any], clusterer) -> Optional[tuple[int, int, int]]:
        """Predict the learner's profile cluster based on their features."""
        if self.learner_profile is not None:
            return self.learner_profile

        
        self.learner_profile = clusterer.predict_student_profile(student_features)
        return self.learner_profile



# Example usage and testing
if __name__ == "__main__":
    print("=== Creating Learner Model ===\n")
    
    # Create a new learner
    learner = LearnerModel(
        learner_id="learner_001",
        name="Alice Johnson"
    )
    
    # Initialize some skills
    learner.add_skill("python", 600)
    learner.add_skill("statistics", 550)
    learner.add_skill("machine learning", 450)
    
    print(f"Learner: {learner.name} (ID: {learner.learner_id})")
    print(f"Initial Skills: {learner.get_all_skills()}\n")
    
    # Define a topic with prerequisites
    topic = {
        'name': "Introduction to Machine Learning",
        'prerequisites': {
            'python': 500,
            'statistics': 600,
            'machine learning': 500
        }
    }
    
    print(f"=== Checking Readiness for '{topic['name']}' ===\n")
    readiness = learner.check_readiness_for_topic(
        topic['name'],
        topic['prerequisites']
    )
    
    print(f"Overall Readiness: {readiness['overall_readiness']:.2%}\n")
    print("Skill Analysis:")
    for skill, match_info in readiness['skill_matches'].items():
        gap = readiness['skill_gaps'][skill]
        print(f"  {skill}:")
        print(f"    Current: {match_info['learner_mastery']:.0f} | "
              f"Required: {match_info['required_mastery']:.0f} | "
              f"Gap: {gap:+.0f}")
    
    print(f"\nReady Skills: {readiness['ready_skills']}")
    print(f"Weak Skills: {readiness['weak_skills']}\n")
    
    # Simulate topic completion
    print("=== Completing Topic ===\n")
    print("Learner completes topic with 75% performance in 3600 seconds")
    
    updated_skills = learner.record_topic_completion(
        content_id="ml_intro_001",
        topic_name=topic['name'],
        topic_prerequisites=topic['prerequisites'],
        performance_score=0.75,
        duration_seconds=3600,
        engagement_score=0.85
    )
    
    print("\nSkill Updates:")
    for skill, new_mastery in updated_skills.items():
        old_mastery = readiness['skill_matches'][skill]['learner_mastery']
        change = new_mastery - old_mastery
        print(f"  {skill}: {old_mastery:.0f} → {new_mastery:.0f} ({change:+.0f})")
    
    # Check readiness again
    print("\n=== Re-evaluation After Completion ===\n")
    new_readiness = learner.check_readiness_for_topic(
        topic['name'],
        topic['prerequisites']
    )
    
    print(f"New Overall Readiness: {new_readiness['overall_readiness']:.2%}")
    print(f"Readiness Improvement: {new_readiness['overall_readiness'] - readiness['overall_readiness']:+.2%}\n")
    
    # Show statistics
    print("=== Learning Statistics ===\n")
    stats = learner.get_learning_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test serialization
    print("\n=== Testing Serialization ===\n")
    learner.save_to_json("/tmp/test_learner.json")
    print("Saved to /tmp/test_learner.json")
    
    loaded_learner = LearnerModel.load_from_json("/tmp/test_learner.json")
    print(f"Loaded learner: {loaded_learner.name}")
    print(f"Skills match: {loaded_learner.get_all_skills() == learner.get_all_skills()}")