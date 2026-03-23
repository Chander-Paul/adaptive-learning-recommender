"""
Main recommender engine.

Uses embeddings generating from content model, learner profiles generated from cluster preictions to recommend learning resources

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
from helper_functions import map_resource_medium_to_type



class ResourceRecommender:
    """
    Recommends resources using multi-view learner types and lightweight rules.

    Inputs:
    - resources_path: CSV file containing resource information (id, title, url, topic_name, author, medium, date)
    - multiview_profiles_path: CSV file containing learner profiles with cluster assignments
    - embeddings_mapping_path: NPY file containing mapping of resource IDs to their embedding vectors
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
        resources_path: str = "data/resources_with_bertopic.csv",
        topics_path: str = "data/berttopic_topics.csv",
        embeddings_mapping_path: str = "embeddings/all-MiniLM-L6-v2_resources.csv_embeddings_mapping.npy",

    ) -> None:
        
        ## Load resources and premade embeddings and suer profiles.
        self.resources_path = resources_path

        self.resources = pd.read_csv(resources_path)
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


    def recommend(self, profile=None, top_n: int = 10, query: Optional[str] = None,
    filter: list[dict[str, Any]] = None,
    similar_to_content_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Return top_n resources for the given learner profile."""
        
        #composite = self.infer_composite_profile(profile)
        #profile_info = self.get_profile_info(composite)


        ## Generate candidates from similarity query
        content = ContentModel()
        content.load_model()
        ## switch to database connection for performance 
        content.embeddings = content.load_embeddings(content.embedding_file_name)
        
        
        
        candidates = None
        ### Swap between using a user's query, the content they're currently engaged is or their most 
        ### recent content history to generate recommendations.\
        ### Possibly move logic outside of main recommender function.

        if query:
            ## Encoding based on user search input
            query_embedding = content.encode_text(query)
        elif similar_to_content_id:
            ## Encoding based on specific content item. Could be used for "more like this" type recommendations.
            similar_row = self.resources[self.resources["id"] == int(similar_to_content_id)]
            print("Similar Row:", similar_row)
            if not similar_row.empty:
                print("Found similar content for ID:", similar_row.iloc[0])
                query_embedding = content.get_resource_embedding(similar_row.iloc[0])
        elif profile and profile.learner_embeddings is not None:
            ## Encoding based on user profile embedding from engaged content.
            query_embedding = profile.learner_embeddings
        else:
            query_embedding = content.encode_text("TutorialBank")
        
        
        similar_resources = content.get_similar_resources(query_embedding, self.resources, top_k=100)
        candidate_ids = [res[2]["id"] for res in similar_resources]
        candidates = self.resources[self.resources["id"].isin(candidate_ids)].copy()
        
        print("Initial Candidates from similarity search:", len(candidates))

        if filter:
            for f in filter:
                key = f.get("key")
                value = f.get("value")
                if key and value is not None:

                    ## place holder to test type filtering until database has been updated with new column
                    ## allows filtering on raw CSVs by mapping medium -> coarse resource type.
                    if key == "type":
                        value = value.lower()
                        mapped = candidates["medium"].apply(
                            lambda m: (map_resource_medium_to_type(str(m), default=str(m)) or str(m)).lower()
                            if pd.notna(m) else ""
                        )
                        print(f"Candidates before type filter ({value}): {len(candidates)}")
                        candidates = candidates[mapped == value]
                        print(f"Candidates after type filter ({value}): {len(candidates)}")
                        
                    ## Code used when pulling from more stable setup.
                    elif key in candidates.columns:
                        candidates = candidates[candidates[key] == value]
                    else:
                        print(f"Warning: Filter key '{key}' not found in candidates columns.")

        print(f"Candidates after filtering: {len(candidates)}")
        ## Get Candidates from similarity query using users most recent views. 
        if candidates is None or self.ENABLE_PREFERNCES == False:
            candidates = self.apply_preferences(self.resources.copy(), profile)
        
        #if candidates.empty:
        #    candidates = self.resources.copy()

        ## Apply Scoring
        
        candidates['score'] = None
        if self.ENABLE_SCORING:
            if query:
                for idx, row in candidates.iterrows():
                ## Reranking based on query relevance using cross-encoder
                    candidates.loc[idx, "score"] = content.score_text_pair(query, row.get("title", ""))##Scoring using cross-encoder
                     

        ## Rerank recommendations based on user engagement.
        if self.ENABLE_ENGAGEMENT_RERANKING:
            rerank_scored = self.rerank(candidates, profile, top_k=top_n, content=content)
        else:
            rerank_scored = candidates
        
        
        print("Reranked scored candidates:" , len(rerank_scored))
        results = []
        
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
    
    def get_user_topic_mvl_elo_readiness(self, profile: Any, topic: str) -> float:
        """
        Get a user's Multi-Variate ELO score for a given topic based on their skill mastery levels and the topic's prerequisites.
        """
        
        topic_resources = profile.get_all_resources_in_topic(topic)
        topic_mvl_score = len(topic_resources) *500

        return self.cosine_similarity(profile.learner_embeddings, topic_embedding)
    
  

    def get_all_resources_in_topic(self, topic: str) -> pd.DataFrame:
        """
        Get all resources related to a specific topic.
        """
        return self.resources[self.resources["predicted_topic"] == topic]

    def apply_preferences(self, resources: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
        """
        Apply user preferences generated from the learning mode clustering model.
        
        
        """

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
    


    
    def score_resource(self, 
                       resource: pd.Series,
                       profile_info: dict[str, Any],
                       profile: dict[str, Any],
                       ) -> float:
        
        """
        Score recommendations absed on user profile attributes.
        """
        

        
        score = 1.0
        medium = str(resource.get("medium", ""))
        date = self.safe_date(resource)

        style = profile_info.get("LearningStyle")
        performance = profile_info.get("Performance")
        affective = profile_info.get("Affective")

        print("Check For Style")
        print(style)
        if style and medium in self.STYLE_MEDIUM_BOOST.get(style, set()):
            score += 0.4
        if performance and medium in self.PERFORMANCE_MEDIUM_BOOST.get(performance, set()):
            score += 0.3
        if affective == "Moderately Engaged" and medium in {"Tutorials", "Resources"}:
            score += 0.1
        if affective == "Disengaged" and date is not None and date >= 2015:
            score += 0.1

        content_sim = self.content_similarity(resource, profile)
        score += content_sim * self.CONTENT_SIM_WEIGHT

        return score
    
    def rerank(
        self,
        candidates: list[tuple[float, pd.Series]],
        profile: Any,    
        top_k: int,
        content: ContentModel = None,
    ) -> list[tuple[float, pd.Series]]:


        


        if top_k <= 1 or len(candidates)<= 1:
            return candidates   
        engaged_content = profile.get_engaged_content()
        #if not engaged_content:
        #    return candidates
        head = candidates[:top_k]
        tail = candidates[len(candidates)-top_k:]

        reranked = []
        ## Add scoring based on similarity to engaged content

        for idx, row in head.iterrows():
            base_score = float(row["score"]) if pd.notna(row.get("score")) else 0.0
            sim = self.user_content_similarity(row, profile, content)
            
            reranked_score = base_score + (sim * self.RERANK_SIM_WEIGHT)

            head.loc[idx, "score"] = reranked_score
            reranked.append((reranked_score, head.loc[[idx]]))
        
        ## Rerank based on learner MVL Elo score
        


        reranked.sort(key=lambda x: (x[0], self.safe_date(x[1].iloc[0])), reverse=True)
        top_rows = pd.concat([r for _, r in reranked[:top_k]])
        print("Reranked top candidates:" , len(top_rows))

        return top_rows
    



    def user_content_similarity(self, resource: pd.Series, profile: Any, content: Any) -> float:

        resource_embedding = content.get_resource_embedding(resource)
        if resource_embedding is None:
            return 0.0
        if profile.learner_embeddings is None:
            return 0.0
        return self.cosine_similarity(profile.learner_embeddings, resource_embedding)







    def load_embeddings(self) -> None:
        if self.resource_embeddings is not None:
            return
        if not os.path.exists(self.embeddings_mapping_path):
            print(f"Embeddings mapping file not found at {self.embeddings_mapping_path}")
            return

        mapping = np.load(self.embeddings_mapping_path, allow_pickle=True)
        embeddings = {}
        for item in mapping[:10]:
            try:
                resource_id, embedding = item
                
                ##print(f"Loading embedding for item: {resource_id}")
                embeddings[int(resource_id)] = np.asarray(embedding, dtype=float)
            except Exception:
                continue

        self.resource_embeddings = embeddings

   
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def safe_date(resource: pd.Series) -> Optional[int]:
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