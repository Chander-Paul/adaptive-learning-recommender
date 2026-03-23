from datetime import datetime, timezone
from typing import Any

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship


db = SQLAlchemy()
migrate = Migrate()

content_topics = db.Table(
    "content_topics",
    db.Column("content_id", db.Integer, db.ForeignKey("content.id"), primary_key=True),
    db.Column("topic_id", db.Integer, db.ForeignKey("topics.id"), primary_key=True),
)

skill_topic_mapping = db.Table(
    "skill_topic_mapping",
    db.Column("skill_id", db.Integer, db.ForeignKey("skills.id"), primary_key=True),
    db.Column("topic_id", db.Integer, db.ForeignKey("topics.id"), primary_key=True),
)


class Content(db.Model):
    """Database table for content as referenced by interactions."""

    __tablename__ = "content"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_id: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resource_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    content_metadata: Mapped[dict[str, Any]] = mapped_column(db.JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    interaction_rows: Mapped[list["ContentInteractions"]] = relationship(
        back_populates="content",
        cascade="all, delete-orphan",
    )
    topics: Mapped[list["Topic"]] = relationship(
        "Topic",
        secondary=content_topics,
        back_populates="content_rows",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content_id": self.content_id,
            "topic_id": self.topics[0].id if self.topics else None,
            "topic_name": self.topics[0].topic_id if self.topics else None,
            "topic_ids": [topic.id for topic in self.topics],
            "topic_names": [topic.topic_id for topic in self.topics],
            "title": self.title,
            "resource_type": self.resource_type,
            "content_metadata": self.content_metadata,
            "created_at": self.created_at.isoformat(),
        }


class Topic(db.Model):
    """Topic table with a self-referential prerequisite hierarchy."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    topic_metadata: Mapped[dict[str, Any]] = mapped_column(db.JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    prerequisite_topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id"),
        nullable=True,
        index=True,
    )

    prerequisite_topic: Mapped["Topic | None"] = relationship(
        "Topic",
        remote_side=[id],
        back_populates="dependent_topics",
    )
    dependent_topics: Mapped[list["Topic"]] = relationship(
        "Topic",
        back_populates="prerequisite_topic",
    )
    content_rows: Mapped[list["Content"]] = relationship(
        "Content",
        secondary=content_topics,
        back_populates="topics",
    )
    skills: Mapped[list["Skill"]] = relationship(
        "Skill",
        secondary=skill_topic_mapping,
        back_populates="topics",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic_id": self.topic_id,
            "title": self.title,
            "prerequisite_topic_id": self.prerequisite_topic_id,
            "prerequisite_topic_name": self.prerequisite_topic.topic_id if self.prerequisite_topic else None,
            "topic_metadata": self.topic_metadata,
            "created_at": self.created_at.isoformat(),
        }


class LearnerModels(db.Model):
    """Database table for persisted learner model snapshots."""

    __tablename__ = "learner_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_id: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=True, default="")
    registration_date: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    default_mastery: Mapped[float] = mapped_column(Float, nullable=False, default=500.0)
    skills: Mapped[dict[str, Any]] = mapped_column(db.JSON, nullable=False, default=dict) ## Treat skills as topics
    content_interactions: Mapped[list[dict[str, Any]]] = mapped_column(
        db.JSON,
        nullable=False,
        default=list,
    )
    total_learning_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    topics_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_interactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    elo_config: Mapped[dict[str, Any]] = mapped_column(db.JSON, nullable=False, default=dict)

    interaction_rows: Mapped[list["ContentInteractions"]] = relationship(
        back_populates="learner",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "learner_id": self.learner_id,
            "name": self.name,
            "registration_date": self.registration_date.isoformat(),
            "default_mastery": self.default_mastery,
            "skills": self.skills,
            "content_interactions": self.content_interactions,
            "total_learning_time": self.total_learning_time,
            "topics_completed": self.topics_completed,
            "total_interactions": self.total_interactions,
            "elo_config": self.elo_config,
        }


class Skill(db.Model):
    """Skill table for reusable skill entities."""

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    skill_id: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    prerequisite_skill_id: Mapped[int | None] = mapped_column(
        ForeignKey("skills.id"),
        nullable=True,
        index=True,
    )

    prerequisite_skill: Mapped["Skill | None"] = relationship(
        "Skill",
        remote_side=[id],
        back_populates="dependent_skills",
    )
    dependent_skills: Mapped[list["Skill"]] = relationship(
        "Skill",
        back_populates="prerequisite_skill",
    )
    topics: Mapped[list["Topic"]] = relationship(
        "Topic",
        secondary=skill_topic_mapping,
        back_populates="skills",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill_id": self.skill_id,
            "title": self.title,
            "prerequisite_skill_id": self.prerequisite_skill_id,
            "prerequisite_skill_name": self.prerequisite_skill.skill_id if self.prerequisite_skill else None,
            "topic_ids": [topic.id for topic in self.topics],
            "topic_names": [topic.topic_id for topic in self.topics],
            "skill_metadata": self.skill_metadata,
            "created_at": self.created_at.isoformat(),
        }


class ContentInteractions(db.Model):
    """Database table for learner content interactions."""

    __tablename__ = "content_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    learner_model_id: Mapped[int] = mapped_column(
        ForeignKey("learner_models.id"),
        nullable=False,
        index=True,
    )
    content_record_id: Mapped[int] = mapped_column(
        ForeignKey("content.id"),
        nullable=False,
        index=True,
    )
    content_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id"),
        nullable=True,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    performance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    skills_practiced: Mapped[list[str]] = mapped_column(db.JSON, nullable=False, default=list)
    feedback: Mapped[int | None] = mapped_column(Integer, nullable=True)

    learner: Mapped["LearnerModels"] = relationship(back_populates="interaction_rows")
    content: Mapped["Content"] = relationship(back_populates="interaction_rows")
    topic: Mapped["Topic | None"] = relationship()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "learner_model_id": self.learner_model_id,
            "content_record_id": self.content_record_id,
            "content_id": self.content_id,
            "learner_id": self.learner.learner_id if self.learner else None,
            "topic_id": self.topic_id,
            "topic_name": self.topic.topic_id if self.topic else None,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "completed": self.completed,
            "performance_score": self.performance_score,
            "engagement_score": self.engagement_score,
            "skills_practiced": self.skills_practiced,
            "feedback": self.feedback,
        }

