"""
Simple learner-type aware recommender.

Uses multi-view clustering outputs to infer a composite learner profile and
prioritizes resources by a small set of profile-aware heuristics.
"""

from __future__ import annotations

import json
import os
import pathlib
import select
from typing import Any, Optional

import numpy as np
import pandas as pd
from models.content.content_model import ContentModel



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
        
        ## Load resources and premade embeddings and suer profiles.
        self.resources_path = resources_path

        self.resources = pd.read_csv(resources_path)
        self.multiview_profiles_path = multiview_profiles_path
        self.multiview_recommendations_path = multiview_recommendations_path
        self.embeddings_mapping_path = embeddings_mapping_path

        self.profiles_df = None
        self.recommendations = None
        self.resource_embeddings = None
        self.topic_embeddings = None


        ## Settings to allow for testing each models effect on recommendations
        self.ENABLE_PREFERNCES = True
        self.ENABLE_USER_CONTENT_SIMILARITY = True
        self.ENABLE_ENGAGEMENT_RERANKING = True
        self.ENABLE_SCORING = True


    def recommend(self, profile: dict[str, Any], top_n: int = 10, query: Optional[str] = None) -> list[dict[str, Any]]:
        """Return top_n resources for the given learner profile."""
        
        #composite = self.infer_composite_profile(profile)
        #profile_info = self.get_profile_info(composite)


        ## Generate candidates from similarity query
        content = ContentModel()
        content.load_model()
        ## switch to database connection for performance 
        content.embeddings = content.load_embeddings(content.embedding_file_name)
        
        
        
        candidates = None
        if query:
            ## Encoding based on user search input
            query_embedding = content.encode_text(query)
        else:
            ## Encoding based on user profile embedding from engaged content.
            query_embedding = profile.learner_embeddings
        similar_resources = content.get_similar_resources(query_embedding, self.resources, top_k=100)
        candidate_ids = [res[2]["id"] for res in similar_resources]
        candidates = self.resources[self.resources["id"].isin(candidate_ids)].copy()
        


        ## Get Candidates from similarity query using users most recent views. 
 
        
        []
        if candidates.empty:
            candidates = self.resources.copy()

        print(candidates)
        ## Apply Scoring
        
        candidates['score'] = None
        if self.ENABLE_SCORING:
            if query:
                for idx, row in candidates.iterrows():
                ## Reranking based on query relevance using cross-encoder
                    candidates.loc[idx, "score"] = content.score_text_pair(query, row.get("title", ""))##Scoring using cross-encoder
                     

        ## Rerank recommendations based on user engagement.
        if self.ENABLE_ENGAGEMENT_RERANKING:
            rerank_scored = self.rerank(candidates, profile, top_k=self.RERANK_TOP_K)
        else:
            rerank_scored = candidates
        
        
        print("Reranked scored candidates:")
        # print(rerank_scored)
        results = []
        print(type(rerank_scored))
        for _, row in rerank_scored.iterrows():
            

            
            results.append(
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "topic": row.get("topic_name"),
                    "author": row.get("author"),
                    "medium": row.get("medium"),
                    "date": row.get("date"),
                    "score": row.get("score"),
                    "composite_profile": "test_profile",
                    "url": row.get("url")
                }
            )
        return results

    def score_resource_for_learner(self, resource: pd.Series, profile: dict[str, Any]) -> float:
        """Score a resource for a learner based on profile and resource features."""
        score = 0.0
        
        ## Boost score based on medium preferences from inferred learner type.
        if self.ENABLE_PREFERNCES:
            preferred_mediums = self.STYLE_MEDIUM_BOOST.get(profile.get("style"), set()) | self.PERFORMANCE_MEDIUM_BOOST.get(profile.get("performance"), set())
            if resource.get("medium") in preferred_mediums:
                score += 1.0
        ## Boost score based on content similarity to recently engaged content.
        if self.ENABLE_USER_CONTENT_SIMILARITY and profile.get("learner_embeddings") is not None:
            resource_embedding = self.resource_embeddings.get(resource["id"])
            if resource_embedding is not None:
                sim = np.dot(profile["learner_embeddings"], resource_embedding) / (np.linalg.norm(profile["learner_embeddings"]) * np.linalg.norm(resource_embedding) + 1e-10)
                score += self.CONTENT_SIM_WEIGHT * sim  
                