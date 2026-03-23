from datetime import datetime, timezone

from flask import Flask, jsonify, request

try:
    from .db_models import Content, ContentInteractions, LearnerModels, Topic, db
except ImportError:  # pragma: no cover - fallback for direct script-style execution
    from db_models import Content, ContentInteractions, LearnerModels, Topic, db


def register_routes(app: Flask) -> None:
    """Attach CLI and API routes to the Flask app instance."""

    @app.cli.command("init-db")
    def init_db_command() -> None:
        with app.app_context():
            db.create_all()
        print("Database initialized.")

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
    

