from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, render_template, request

try:
    from .db_models import Content, ContentInteractions, LearnerModels, Topic, db
except ImportError:  # pragma: no cover - fallback for direct script-style execution
    from db_models import Content, ContentInteractions, LearnerModels, Topic, db


def register_routes(app: Flask) -> None:
    """Attach CLI and API routes to the Flask app instance."""

    # Instantiated once at startup so model weights are not reloaded per-request.
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from recommender import ResourceRecommender
    from adaptive_recommender import AdaptiveRecommender
    from models.learner.learner_model import LearnerModel, ContentInteraction

    resource_recommender = ResourceRecommender()

    ### Adaptive Resource Recommender suddenly stopped working when loaded from the API. Works when loaded from tests.
    ### Gives errors when attempting to load model details. 
    #adaptive_recommender = AdaptiveRecommender(model_path="models/learning_mode_lstm_model.keras")

    def build_learner_model(payload: dict[str, Any]) -> LearnerModel:
        """Reconstruct a LearnerModel from a request payload."""
        profile = LearnerModel(
            learner_id=str(payload["learner_id"]),
            name=payload.get("learner_name", ""),
        )
        for raw in payload.get("interactions", []):
            profile.log_content_interaction(
                ContentInteraction(
                    content_id=str(raw["content_id"]),
                    topic_id=raw.get("topic_id"),
                    timestamp=raw.get("timestamp"),
                    duration_seconds=int(raw.get("duration_seconds", 0)),
                    completed=bool(raw.get("completed", False)),
                    performance_score=raw.get("performance_score"),
                    engagement_score=raw.get("engagement_score"),
                    feedback=raw.get("feedback"),
                )
            )
        return profile

    @app.cli.command("init-db")
    def init_db_command() -> None:
        with app.app_context():
            db.create_all()
        print("Database initialized.")

    @app.get("/")
    def recommender_spa() -> str:
        return render_template("recommender_spa.html")



    @app.post("/0")
    def create_interaction():
        payload = request.get_json(silent=True) or {}

        if "learner_id" not in payload or "content_id" not in payload:
            return jsonify({"error": "learner_id and content_id are required"}), 400

        learner = db.session.query(LearnerModels).filter_by(
            learner_id=payload["learner_id"]
        ).one_or_none()
        if learner is None:
            learner = LearnerModels(
                learner_id=payload["learner_id"],
                name=payload.get("learner_name", ""),
            )
            db.session.add(learner)
            db.session.flush()

        content = db.session.query(Content).filter_by(
            content_id=payload["content_id"]
        ).one_or_none()

        topic_names: list[str] = []
        if payload.get("topic_id"):
            topic_names.append(str(payload["topic_id"]))

        provided_topic_ids = payload.get("topic_ids")
        if isinstance(provided_topic_ids, list):
            topic_names.extend(str(topic_name) for topic_name in provided_topic_ids if topic_name)

        topic_names = list(dict.fromkeys(topic_names))

        topics: list[Topic] = []
        for topic_name in topic_names:
            topic = db.session.query(Topic).filter_by(topic_id=topic_name).one_or_none()
            if topic is None:
                prerequisite_name = payload.get("prerequisite_topic_id")
                prerequisite = None
                if prerequisite_name:
                    prerequisite = db.session.query(Topic).filter_by(topic_id=prerequisite_name).one_or_none()
                topic = Topic(
                    topic_id=topic_name,
                    title=payload.get("topic_title"),
                    prerequisite_topic_id=prerequisite.id if prerequisite else None,
                    topic_metadata=payload.get("topic_metadata", {}),
                )
                db.session.add(topic)
                db.session.flush()
            topics.append(topic)

        if content is None:
            content = Content(
                content_id=payload["content_id"],
                title=payload.get("content_title"),
                resource_type=payload.get("resource_type"),
                content_metadata=payload.get("content_metadata", {}),
            )
            db.session.add(content)
            db.session.flush()

        if topics:
            existing_topic_ids = {topic.id for topic in content.topics}
            for topic in topics:
                if topic.id not in existing_topic_ids:
                    content.topics.append(topic)

        primary_topic = topics[0] if topics else None

        row = ContentInteractions(
            learner_model_id=learner.id,
            content_record_id=content.id,
            content_id=payload["content_id"],
            topic_id=primary_topic.id if primary_topic else None,
            timestamp=datetime.fromisoformat(payload["timestamp"]) if payload.get("timestamp") else datetime.now(timezone.utc),
            duration_seconds=int(payload.get("duration_seconds", 0)),
            completed=bool(payload.get("completed", False)),
            performance_score=payload.get("performance_score"),
            engagement_score=payload.get("engagement_score"),
            skills_practiced=payload.get("skills_practiced", []),
            feedback=payload.get("feedback"),
        )

        db.session.add(row)
        db.session.commit()

        return jsonify(row.to_dict()), 201

    @app.get("/interactions")
    def list_interactions():
        limit = request.args.get("limit", default=50, type=int)
        rows = (
            db.session.query(ContentInteractions)
            .order_by(ContentInteractions.timestamp.desc())
            .limit(max(1, min(limit, 500)))
            .all()
        )
        return jsonify([row.to_dict() for row in rows])

    @app.get("/topics")
    def list_topics():
        topics = db.session.query(Topic).all()
        return jsonify([{"id": topic.id, "topic_id": topic.topic_id, "title": topic.title} for topic in topics])
    
    @app.post("/topics")
    def create_topic():
        payload = request.get_json(silent=True) or {}
        if "topic_id" not in payload:
            return jsonify({"error": "topic_id is required"}), 400

        existing_topic = db.session.query(Topic).filter_by(topic_id=payload["topic_id"]).one_or_none()
        if existing_topic:
            return jsonify({"error": "Topic with this topic_id already exists"}), 400

        prerequisite_name = payload.get("prerequisite_topic_id")
        prerequisite = None
        if prerequisite_name:
            prerequisite = db.session.query(Topic).filter_by(topic_id=prerequisite_name).one_or_none()
            if prerequisite is None:
                return jsonify({"error": "Prerequisite topic not found"}), 400

        topic = Topic(
            topic_id=payload["topic_id"],
            title=payload.get("title"),
            prerequisite_topic_id=prerequisite.id if prerequisite else None,
            topic_metadata=payload.get("topic_metadata", {}),
        )
        db.session.add(topic)
        db.session.commit()

        return jsonify({"id": topic.id, "topic_id": topic.topic_id, "title": topic.title}), 201
    
    @app.get("/learners")
    def list_learners():
        learners = db.session.query(LearnerModels).all()
        return jsonify([{"id": learner.id, "learner_id": learner.learner_id, "name": learner.name} for learner in learners])
    
    @app.get("/content")
    def list_content():
        content_items = db.session.query(Content).all()
        return jsonify([{"id": content.id, "content_id": content.content_id, "title": content.title} for content in content_items])

    @app.post("/recommend")
    def get_recommendations():
        """
        Get recommendations for a learner.

        Expected JSON body:
        {
            "learner_id": "1",
            "learner_name": "Jane Doe",          // optional
            "interactions": [                     // optional – recent content history
                {
                    "content_id": "11",
                    "topic_id": "1",              // optional
                    "timestamp": "2024-01-01T10:00:00Z",
                    "duration_seconds": 300,
                    "feedback": 4,
                    "completed": false
                }
            ],
            "top_n": 10,                          // optional, default 10
            "query": "...",                       // optional – free-text search
            "filter": [{"key": "type", "value": "resource"}],  // optional
            "similar_to_content_id": "11"         // optional – 'more like this'
        }
        """
        payload = request.get_json(silent=True) or {}

        if "learner_id" not in payload:
            return jsonify({"error": "learner_id is required"}), 400

        profile = build_learner_model(payload)

        results = resource_recommender.recommend(
            profile=profile,
            top_n=int(payload.get("top_n", 10)),
            query=payload.get("query"),
            filter=payload.get("filter"),
            similar_to_content_id=payload.get("similar_to_content_id"),
        )

        return jsonify({"recommendations": results}), 200

    @app.post("/recommend/adaptive")
    def get_adaptive_recommendations():
        """
        Get recommendations personalised to the learner's current learning mode,
        predicted from their recent in-session action sequence.

        Expected JSON body:
        {
            "learner_id": "1",
            "learner_name": "Jane Doe",           // optional
            "interactions": [...],                 // optional – same format as /recommend
            "recent_actions": [["enter","q"], ["play_video","l"], ...],  // required
            "top_n": 5,                            // optional, default 10
            "triggered_resource": false,           // optional
            "trigger_content_id": "11"             // optional – required when triggered_resource=true
        }
        """
        payload = request.get_json(silent=True) or {}

        if "learner_id" not in payload:
            return jsonify({"error": "learner_id is required"}), 400

        recent_actions = payload.get("recent_actions")
        if not recent_actions:
            return jsonify({"error": "recent_actions is required"}), 400

        profile = build_learner_model(payload)

        recommendations, predicted_mode = adaptive_recommender.recommend(
            profile=profile,
            recent_actions=recent_actions,
            top_n=int(payload.get("top_n", 10)),
            triggered_resource=bool(payload.get("triggered_resource", False)),
            trigger_content_id=payload.get("trigger_content_id"),
        )

        return jsonify({
            "recommendations": recommendations,
            "predicted_mode": predicted_mode,
        }), 200
    

