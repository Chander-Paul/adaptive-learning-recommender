"""Class for managing a learner object
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json
from multiview_clustering import MultiViewLearningPathClusterer
from mvl_elo import MultiVariateEloTopicMastery


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'skill_name': self.skill_name,
            'mastery_score': self.mastery_score,
            'last_updated': self.last_updated.isoformat(),
            'history': self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillMastery':
        """Create SkillMastery from dictionary."""
        return cls(
            skill_name=data['skill_name'],
            mastery_score=data['mastery_score'],
            last_updated=datetime.fromisoformat(data['last_updated']),
            history=data.get('history', [])
        )


@dataclass
class ContentInteraction:
    """
        Represents a learner's interaction with learning content
    """
    
    content_id: str
    topic_name: str
    timestamp: datetime
    duration_seconds: int
    completed: bool
    performance_score: Optional[float] = None
    engagement_score: Optional[float] = None
    skills_practiced: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content_id': self.content_id,
            'topic_name': self.topic_name,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'completed': self.completed,
            'performance_score': self.performance_score,
            'engagement_score': self.engagement_score,
            'skills_practiced': self.skills_practiced
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentInteraction':
        """Create ContentInteraction from dictionary."""
        return cls(
            content_id=data['content_id'],
            topic_name=data['topic_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            duration_seconds=data['duration_seconds'],
            completed=data['completed'],
            performance_score=data.get('performance_score'),
            engagement_score=data.get('engagement_score'),
            skills_practiced=data.get('skills_practiced', [])
        )

@dataclass

class StudyHabits:
    """Represents a learner's study habits and preferences."""
    
    preferred_study_times: List[str] = field(default_factory=list)  # e.g., ["morning", "evening"]
    average_session_length_minutes: Optional[float] = None
    preferred_content_types: List[str] = field(default_factory=list)  # e.g., audio, visual, kinesthetic
    study_environment: Optional[str] = None  # dsicussion, online, selfstudy
    weekly_study_hours: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudyHabits':
        """Create StudyHabits from dictionary."""
        return cls(
            preferred_study_times=data.get('preferred_study_times', []),
            average_session_length_minutes=data.get('average_session_length_minutes'),
            preferred_content_types=data.get('preferred_content_types', []),
            study_environment=data.get('study_environment')
        )

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
        # Statistics
        self.total_learning_time = 0  # in seconds
        self.topics_completed = 0
        self.total_interactions = 0
    
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
    
    def record_topic_completion(self,content_id: str,topic_name: str,topic_prerequisites: Dict[str, float],
        performance_score: float,duration_seconds: int,engagement_score: Optional[float] = None,completed: bool = True
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
    
    def get_recent_interactions(self, limit: int = 10) -> List[ContentInteraction]:
        """Get the most recent content interactions."""
        return sorted(
            self.content_interactions,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the learner model to a dictionary for serialization."""
        return {
            'learner_id': self.learner_id,
            'name': self.name,
            'registration_date': self.registration_date.isoformat(),
            'default_mastery': self.default_mastery,
            'skills': {name: skill.to_dict() for name, skill in self.skills.items()},
            'content_interactions': [i.to_dict() for i in self.content_interactions],
            'total_learning_time': self.total_learning_time,
            'topics_completed': self.topics_completed,
            'total_interactions': self.total_interactions,
            'elo_config': {
                'k_factor': self.elo_matcher.k_factor,
                'default_mastery': self.elo_matcher.default_mastery,
                'scaling_factor': self.elo_matcher.scaling_factor
            }
        }
    
    def save_to_json(self, filepath: str):
        """Save the learner model to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearnerModel':
        """Create a LearnerModel from a dictionary."""
        elo_config = data.get('elo_config', {})
        elo_matcher = MultiVariateEloTopicMastery(
            k_factor=elo_config.get('k_factor', 32),
            default_mastery=elo_config.get('default_mastery', 500),
            scaling_factor=elo_config.get('scaling_factor', 400)
        )
        
        learner = cls(
            learner_id=data['learner_id'],
            name=data.get('name', ''),
            elo_matcher=elo_matcher,
            default_mastery=data.get('default_mastery', 500)
        )
        
        learner.registration_date = datetime.fromisoformat(data['registration_date'])
        learner.total_learning_time = data.get('total_learning_time', 0)
        learner.topics_completed = data.get('topics_completed', 0)
        learner.total_interactions = data.get('total_interactions', 0)
        
        # Restore skills
        for skill_name, skill_data in data.get('skills', {}).items():
            learner.skills[skill_name] = SkillMastery.from_dict(skill_data)
        
        # Restore interactions
        for interaction_data in data.get('content_interactions', []):
            learner.content_interactions.append(
                ContentInteraction.from_dict(interaction_data)
            )
        
        return learner
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'LearnerModel':
        """Load a learner model from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    

    def get_learning_path_cluster(self) -> List[Dict[str, Any]]:
        """Get the learner's learning path based on content interactions."""
        from multiview_clustering import MultiViewLearningPathClusterer

        multiview = MultiViewLearningPathClusterer(
            data_path='data/training_files/student_performance.csv',
            use_minmax_scaler=True  # Keep all values positive
            



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