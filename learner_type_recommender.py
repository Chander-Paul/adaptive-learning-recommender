"""
Simple learner-type aware recommender.

Uses multi-view clustering outputs to infer a composite learner profile and
prioritizes resources by a small set of profile-aware heuristics.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from content_model import ContentModel


class ResourceRecommender:
    """
    Recommends resources using multi-view learner types and lightweight rules.

    Inputs:
    - resources-v2023-clean.tsv (content catalog)
    - multiview_student_profiles.csv (cluster assignments)
    - multiview_recommendations.json (cluster interpretations)
    """

    STYLE_MEDIUM_BOOST = {
        "Digital Self-Learner": {"Libraries", "Resources", "Tutorials"},
        "Collaborative Learner": {"Tutorials", "Resources"},
        "Deep Independent Learner": {"Surveys", "Tutorials"},
        "Traditional Learner": {"Tutorials", "Surveys"},
    }

    PERFORMANCE_MEDIUM_BOOST = {
        "High Performer": {"Surveys", "Tutorials"},
        "Moderate Performer": {"Tutorials", "Resources"},
        "Struggling Performer": {"Tutorials", "Resources"},
    }

    CONTENT_SIM_WEIGHT = 0.6
    RERANK_TOP_K = 40
    RERANK_SIM_WEIGHT = 0.3

    def __init__(
        self,
        resources_path: str = "data/resources.csv",
        multiview_profiles_path: str = "visualizations/multiview_student_profiles.csv",
        multiview_recommendations_path: str = "visualizations/multiview_recommendations.json",
        embeddings_mapping_path: str = "embeddings/all-MiniLM-L6-v2_resources.csv_embeddings_mapping.npy",
    ) -> None:
        self.resources_path = resources_path
        self.multiview_profiles_path = multiview_profiles_path
        self.multiview_recommendations_path = multiview_recommendations_path
        self.embeddings_mapping_path = embeddings_mapping_path

        self.resources = pd.read_csv(self.resources_path)
        self.profiles_df = None
        self.recommendations = None
        self.resource_embeddings = None
        self.topic_embeddings = None

    def recommend(self, profile: dict[str, Any], top_n: int = 10, query: Optional[str] = None) -> list[dict[str, Any]]:
        """Return top_n resources for the given learner profile."""
        composite = self.infer_composite_profile(profile)
        profile_info = self.get_profile_info(composite)

        candidates = None
        if query:
            content = ContentModel()
            content.load_model()
            content.embeddings = content.load_embeddings(content.embedding_file_name)
            query_embedding = content.encode_text(query)
            similar_resources = content.get_similar_resources(query_embedding, self.resources, top_k=100)
            candidate_ids = [res[2]["id"] for res in similar_resources]
            candidates = self.resources[self.resources["id"].isin(candidate_ids)]

        if candidates is None:
            candidates = self.apply_preferences(self.resources.copy(), profile)
        if candidates.empty:
            candidates = self.resources.copy()

        ## Apply Scoring
        scored = []
        for _, row in candidates.iterrows():
            score = self.score_resource(row, profile_info, profile)
            scored.append((score, row))
        scored.sort(key=lambda x: (x[0], self._safe_date(x[1])), reverse=True)

        ## Rerank recommendations based on user engagement.
        scored = self.rerank_by_engagement(scored, profile, top_k=self.RERANK_TOP_K)

        results = []
        for score, row in scored[:top_n]:
            results.append(
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "topic": row.get("topic_name"),
                    "author": row.get("author"),
                    "medium": row.get("medium"),
                    "date": row.get("date"),
                    "score": round(float(score), 4),
                    "composite_profile": composite,
                    "url": row.get("url")
                }
            )

        return results

    def apply_preferences(self, resources: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
        preferences = profile.get("preferences", {})
        if not preferences:
            return resources

        mediums = preferences.get("mediums")
        #Add options for simpler filter options later.
        #min_date = preferences.get("min_date")
        #max_date = preferences.get("max_date")

        filtered = resources
        if mediums:
            allowed = {m.lower() for m in mediums}
            filtered = filtered[
                filtered["medium"].astype(str).str.lower().isin(allowed)
            ]
        #if min_date is not None:
        #    filtered = filtered[pd.notna(filtered["date"]) & (filtered["date"].astype(int) >= int(min_date))]
        #if max_date is not None:
        #    filtered = filtered[pd.notna(filtered["date"]) & (filtered["date"].astype(int) <= int(max_date))]

        return filtered

    def score_resource(
        self,
        resource: pd.Series,
        profile_info: dict[str, Any],
        profile: dict[str, Any],
    ) -> float:
        score = 1.0
        medium = str(resource.get("medium", ""))
        date = self._safe_date(resource)

        style = profile_info.get("LearningStyle")
        performance = profile_info.get("Performance")
        affective = profile_info.get("Affective")

        if style and medium in self.STYLE_MEDIUM_BOOST.get(style, set()):
            score += 0.4
        if performance and medium in self.PERFORMANCE_MEDIUM_BOOST.get(performance, set()):
            score += 0.3
        if affective == "Moderately Engaged" and medium in {"Tutorials", "Resources"}:
            score += 0.1
        if affective == "Disengaged" and date is not None and date >= 2015:
            score += 0.1

        content_sim = self._content_similarity(resource, profile)
        score += content_sim * self.CONTENT_SIM_WEIGHT

        return score

    def infer_composite_profile(self, profile: dict[str, Any]) -> Optional[str]:
        if "Composite_Profile" in profile:
            return str(profile["Composite_Profile"])

        if all(k in profile for k in ["LearningStyle_Cluster", "Performance_Cluster", "Affective_Cluster"]):
            return f"{profile['LearningStyle_Cluster']}_{profile['Performance_Cluster']}_{profile['Affective_Cluster']}"

        self.load_profiles()
        if self.profiles_df is None or self.profiles_df.empty:
            return None

        return self._nearest_profile_match(profile)

    def _nearest_profile_match(self, profile: dict[str, Any]) -> Optional[str]:
        df = self.profiles_df
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
                best_profile = row.get("Composite_Profile")

        return str(best_profile) if best_profile is not None else None

    def get_profile_info(self, composite_profile: Optional[str]) -> dict[str, Any]:
        self.load_recommendations()
        if not composite_profile or not self.recommendations:
            return {}
        return self.recommendations.get(str(composite_profile), {})

    def load_profiles(self) -> None:
        if self.profiles_df is not None:
            return
        if not os.path.exists(self.multiview_profiles_path):
            self.profiles_df = pd.DataFrame()
            return
        self.profiles_df = pd.read_csv(self.multiview_profiles_path)

    def load_recommendations(self) -> None:
        if self.recommendations is not None:
            return
        if not os.path.exists(self.multiview_recommendations_path):
            self.recommendations = {}
            return
        with open(self.multiview_recommendations_path, "r", encoding="utf-8") as f:
            self.recommendations = json.load(f)

    def _content_similarity(self, resource: pd.Series, profile: dict[str, Any]) -> float:
        self._load_embeddings()
        if not self.resource_embeddings:
            return 0.0

        user_embedding = self._build_user_embedding(profile)
        if user_embedding is None:
            return 0.0

        resource_embedding = self._get_resource_embedding(resource)
        if resource_embedding is None:
            return 0.0

        return self._cosine_similarity(user_embedding, resource_embedding)

    def rerank_by_engagement(
        self,
        scored: list[tuple[float, pd.Series]],
        profile: dict[str, Any],
        top_k: int,
    ) -> list[tuple[float, pd.Series]]:
        top_k = max(top_k, 0)
        if top_k <= 0 or not scored:
            return scored
        if not self._has_engagement(profile):
            return scored
        head = scored[:top_k]
        tail = scored[top_k:]

        reranked = []
        for base_score, row in head:
            sim = self._content_similarity(row, profile)
            reranked_score = base_score + (sim * self.RERANK_SIM_WEIGHT)
            reranked.append((reranked_score, row))

        reranked.sort(key=lambda x: (x[0], self._safe_date(x[1])), reverse=True)
        return reranked + tail

    def _build_user_embedding(self, profile: dict[str, Any]) -> Optional[np.ndarray]:
        engaged_ids = self._extract_engaged_resource_ids(profile)
        engaged_topics = self._extract_engaged_topics(profile)

        embeddings = []
        for resource_id in engaged_ids:
            embedding = self.resource_embeddings.get(resource_id)
            if embedding is not None:
                embeddings.append(embedding)

        if engaged_topics:
            self._ensure_topic_embeddings()
            for topic in engaged_topics:
                topic_embedding = self.topic_embeddings.get(topic)
                if topic_embedding is not None:
                    embeddings.append(topic_embedding)

        if not embeddings:
            return None

        return np.mean(np.stack(embeddings, axis=0), axis=0)

    def _get_resource_embedding(self, resource: pd.Series) -> Optional[np.ndarray]:
        resource_id = resource.get("id")
        if pd.isna(resource_id):
            return None
        try:
            resource_id = int(resource_id)
        except Exception:
            return None
        return self.resource_embeddings.get(resource_id)

    def _extract_engaged_resource_ids(self, profile: dict[str, Any]) -> list[int]:
        engaged_ids = profile.get("engaged_resource_ids")
        if isinstance(engaged_ids, list):
            ids = []
            for value in engaged_ids:
                try:
                    ids.append(int(value))
                except Exception:
                    continue
            return ids

        engaged_resources = profile.get("engaged_resources")
        if isinstance(engaged_resources, list):
            ids = []
            for item in engaged_resources:
                if isinstance(item, dict) and "id" in item:
                    try:
                        ids.append(int(item["id"]))
                    except Exception:
                        continue
            return ids

        return []

    def _extract_engaged_topics(self, profile: dict[str, Any]) -> list[str]:
        engaged_topics = profile.get("engaged_topics")
        if isinstance(engaged_topics, list):
            return [str(topic) for topic in engaged_topics if topic]

        engaged_resources = profile.get("engaged_resources")
        if isinstance(engaged_resources, list):
            topics = []
            for item in engaged_resources:
                if isinstance(item, dict):
                    topic = item.get("topic") or item.get("topic_name")
                    if topic:
                        topics.append(str(topic))
            return topics

        return []

    def _has_engagement(self, profile: dict[str, Any]) -> bool:
        if self._extract_engaged_resource_ids(profile):
            return True
        if self._extract_engaged_topics(profile):
            return True
        return False

    def _load_embeddings(self) -> None:
        if self.resource_embeddings is not None:
            return
        if not os.path.exists(self.embeddings_mapping_path):
            self.resource_embeddings = {}
            return

        mapping = np.load(self.embeddings_mapping_path, allow_pickle=True)
        embeddings = {}
        for item in mapping:
            try:
                resource_id, embedding = item
                embeddings[int(resource_id)] = np.asarray(embedding, dtype=float)
            except Exception:
                continue

        self.resource_embeddings = embeddings

    def _ensure_topic_embeddings(self) -> None:
        if self.topic_embeddings is not None:
            return

        self.topic_embeddings = {}
        if not self.resource_embeddings:
            return

        grouped = {}
        for _, row in self.resources.iterrows():
            topic = row.get("topic_name")
            if pd.isna(topic):
                continue
            embedding = self._get_resource_embedding(row)
            if embedding is None:
                continue
            grouped.setdefault(str(topic), []).append(embedding)

        for topic, vectors in grouped.items():
            self.topic_embeddings[topic] = np.mean(np.stack(vectors, axis=0), axis=0)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _safe_date(resource: pd.Series) -> Optional[int]:
        try:
            date = resource.get("date")
            if pd.isna(date):
                return None
            return int(date)
        except Exception:
            return None


if __name__ == "__main__":
    query = "Authorship Matching"
    recommender = ResourceRecommender()

    sample_profile = {
        "StudyHours": 12,
        "OnlineCourses": 6,
        "Discussions": 1,
        "Resources": 1,
        "EduTech": 1,
        "Extracurricular": 0,
        "ExamScore": 78,
        "AssignmentCompletion": 70,
        "Attendance": 82,
        "Motivation": 1,
        "StressLevel": 2,
        "preferences": {"mediums": ["Tutorials", "Resources"], "min_date": 2008},
    }
    
    
    top = recommender.recommend(sample_profile, top_n=5, query=query)
    for i, rec in enumerate(top, 1):
        print(f"{i}. {rec['title']} ({rec['medium']}, {rec['url']}) -> score={rec['score']}")