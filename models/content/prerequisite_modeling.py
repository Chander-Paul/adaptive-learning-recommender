"""
Scripts for prerequisite modelling using bert embeddings.

We will determine prerequisites by identifying topics with
similar but not identical contexts then attempt to identify
a prerequisite relationship based on complexity and one-sided
co-occurence.  
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from content_model import ContentModel
import json
import matplotlib.pyplot as plt


class PrerequisiteGraphGenerator:
    """
    Generates a prerequisite graph of topics using BERT embeddings
    and various heuristics to determine prerequisite relationships.
    """
    
   __init__(self, 
                 content_model: ContentModel = None,
                 resources_path: str = 'data/resources_with_topics_integrated.csv',
                 min_similarity: float = 0.3,
                 max_similarity: float = 0.85,
                 difficulty_weight: float = 0.3):
        """
        
        Args:

        """
        self.content_model = content_model or ContentModel()
        self.resources_path = resources_path
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.difficulty_weight = difficulty_weight
        
        # Graph structure
        self.prerequisite_graph = nx.DiGraph()
        
        # Topic-level data
        self.topic_embeddings: dict[int, np.ndarray] = {}
        self.topic_info: dict[int, dict] = {}
        self.topic_difficulties: dict[int, float] = {}
        
        # Load resources
        self.resources_df = pd.read_csv(resources_path)
        
    def load_embeddings(self, force_regenerate: bool = False):
        """
        Load existing embeddings
        """
        if self.content_model.embedding_model is None:
            self.content_model.load_model()
        
        embedding_file = self.content_model.embedding_file_name
        

        self.content_model.embeddings = self.content_model.load_embeddings(embedding_file)
        # Load mapping
        mapping_file = embedding_file.replace('.npy', '_mapping.npy')
        mapping_data = np.load(self.content_model.embedding_path + mapping_file, allow_pickle=True)
        self.content_model.embeddings_mapping = dict(mapping_data)
        print(f"Loaded {len(self.content_model.embeddings_mapping)} embeddings from cache")
        return   

    def aggregate_topic_embeddings(self):
        """
        Aggregate resource embeddings to topic-level embeddings.
        Uses mean pooling of all resources within a topic.
        """
        print("Aggregating embeddings to topic level...")
        
        # Filter out topics with id = -1 (unclassified)
        valid_topics = self.resources_df[self.resources_df['topic_id'] != -1]
        
        # Group by topic_id
        for topic_id, group in valid_topics.groupby('topic_id'):
            # Get embeddings for all resources in this topic
            topic_resource_embeddings = []
            
            for resource_id in group['id']:
                embedding = self.content_model.get_embedding(resource_id)
                if embedding is not None:
                    topic_resource_embeddings.append(embedding)
            
            if topic_resource_embeddings:
                # Mean pooling
                topic_embedding = np.mean(topic_resource_embeddings, axis=0)
                self.topic_embeddings[int(topic_id)] = topic_embedding
                
                # Store topic metadata
                self.topic_info[int(topic_id)] = {
                    'topic_name': group['topic'].iloc[0] if 'topic' in group.columns else f"Topic {topic_id}",
                    'num_resources': len(group),
                    'resource_ids': group['id'].tolist()
                }
        
        print(f"Created embeddings for {len(self.topic_embeddings)} topics")
    
    def estimate_topic_difficulty(self):
        """
        Estimate difficulty of each topic based on:
        1. Average document complexity (if text available)
        2. Number of resources (more resources = more established/easier)
        3. Topic specificity (measured by embedding variance)
        """
        print("Estimating topic difficulties...")
        
        for topic_id, info in self.topic_info.items():
            difficulty_scores = []
            
            # Factor 1: Document complexity
            complexity_scores = []
            for resource_id in info['resource_ids']:
                resource = self.resources_df[self.resources_df['id'] == resource_id].iloc[0]
                
                # Try to get text content
                if pd.notna(resource.get('txtpath')):
                    try:
                        with open('data/' + resource['txtpath'], 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            if len(text) > 100:  # Only calculate for substantial text
                                complexity = self.content_model.calculate_simple_document_complexity(text)
                                # Invert FK score: lower score = harder to read = higher difficulty
                                # FK scores typically range from 0-100, normalize to 0-1
                                normalized_complexity = max(0, min(1, (100 - complexity) / 100))
                                complexity_scores.append(normalized_complexity)
                    except:
                        pass
            
            if complexity_scores:
                avg_complexity = np.mean(complexity_scores)
                difficulty_scores.append(avg_complexity)
            
            # Factor 2: Resource count (inverse relationship - fewer resources = more specialized/difficult)
            resource_count_score = 1 / (1 + np.log1p(info['num_resources']))
            difficulty_scores.append(resource_count_score)
            
            # Factor 3: Embedding variance (higher variance = more diverse/complex topic)
            topic_resource_embeddings = []
            for resource_id in info['resource_ids']:
                emb = self.content_model.get_embedding(resource_id)
                if emb is not None:
                    topic_resource_embeddings.append(emb)
            
            if len(topic_resource_embeddings) > 1:
                variance = np.var(topic_resource_embeddings)
                # Normalize variance to 0-1 range (use log scale)
                variance_score = min(1, np.log1p(variance * 1000) / 10)
                difficulty_scores.append(variance_score)
            
            # Combine difficulty scores
            if difficulty_scores:
                self.topic_difficulties[topic_id] = np.mean(difficulty_scores)
            else:
                self.topic_difficulties[topic_id] = 0.5  # Default neutral difficulty
        
        print(f"Estimated difficulties for {len(self.topic_difficulties)} topics")
    
    def compute_prerequisite_scores(self) -> Dict[Tuple[int, int], float]:
        """
        Compute prerequisite scores for all topic pairs.
        
        Returns:
            Dictionary mapping (source_topic, target_topic) to prerequisite score
            Higher score = stronger prerequisite relationship
        """
        print("Computing prerequisite scores...")
        
        prerequisite_scores = {}
        topic_ids = list(self.topic_embeddings.keys())
        
        for i, source_topic in enumerate(topic_ids):
            source_embedding = self.topic_embeddings[source_topic]
            source_difficulty = self.topic_difficulties.get(source_topic, 0.5)
            
            for target_topic in topic_ids[i+1:]:  # Avoid duplicate pairs
                if source_topic == target_topic:
                    continue
                
                target_embedding = self.topic_embeddings[target_topic]
                target_difficulty = self.topic_difficulties.get(target_topic, 0.5)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    source_embedding.reshape(1, -1),
                    target_embedding.reshape(1, -1)
                )[0][0]
                
                # Skip if similarity is outside valid range
                if similarity < self.min_similarity or similarity > self.max_similarity:
                    continue
                
                # Determine direction based on difficulty
                # Easier topic should be prerequisite for harder topic
                difficulty_gap = target_difficulty - source_difficulty
                
                # Prerequisite score combines similarity and difficulty gap
                # Positive gap (target harder) increases score
                if abs(difficulty_gap) > 0.05:  # Minimum difficulty difference
                    if difficulty_gap > 0:
                        # source is easier, likely prerequisite for target
                        score = similarity * (1 + self.difficulty_weight * difficulty_gap)
                        prerequisite_scores[(source_topic, target_topic)] = score
                    else:
                        # target is easier, likely prerequisite for source
                        score = similarity * (1 - self.difficulty_weight * difficulty_gap)
                        prerequisite_scores[(target_topic, source_topic)] = score
                else:
                    # Similar difficulty, create bidirectional weak edges
                    score = similarity * 0.5
                    prerequisite_scores[(source_topic, target_topic)] = score
        
        print(f"Computed {len(prerequisite_scores)} potential prerequisite relationships")
        return prerequisite_scores
    
    def build_prerequisite_graph(self, top_k_per_topic: int = 5, min_score: float = 0.4):
        """
        Build the prerequisite DAG from computed scores.
        
        Args:
            top_k_per_topic: Maximum number of prerequisites per topic
            min_score: Minimum score threshold for including an edge
        """
        print("Building prerequisite graph...")
        
        # Compute scores
        prerequisite_scores = self.compute_prerequisite_scores()
        
        # Add nodes
        for topic_id, info in self.topic_info.items():
            self.prerequisite_graph.add_node(
                topic_id,
                name=info['topic_name'],
                difficulty=self.topic_difficulties.get(topic_id, 0.5),
                num_resources=info['num_resources']
            )
        
        # Group edges by target topic to enforce top_k constraint
        edges_by_target = {}
        for (source, target), score in prerequisite_scores.items():
            if score >= min_score:
                if target not in edges_by_target:
                    edges_by_target[target] = []
                edges_by_target[target].append((source, score))
        
        # Add top-k edges for each target topic
        for target, candidates in edges_by_target.items():
            # Sort by score (descending) and take top k
            top_prerequisites = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k_per_topic]
            
            for source, score in top_prerequisites:
                self.prerequisite_graph.add_edge(
                    source,
                    target,
                    weight=score,
                    prerequisite_strength=score
                )
        
        # Ensure graph is acyclic
        self.remove_cycles()
        
        print(f"Built graph with {self.prerequisite_graph.number_of_nodes()} nodes "
              f"and {self.prerequisite_graph.number_of_edges()} edges")
    
    remove_cycles(self):
        """
        Remove cycles from the graph to ensure it's a DAG.
        Removes edges with lowest prerequisite strength first.
        """
        while not nx.is_directed_acyclic_graph(self.prerequisite_graph):
            try:
                # Find a cycle
                cycle = nx.find_cycle(self.prerequisite_graph)
                
                # Find edge with minimum weight in the cycle
                min_edge = min(cycle, key=lambda e: self.prerequisite_graph[e[0]][e[1]]['weight'])
                
                # Remove the weakest edge
                self.prerequisite_graph.remove_edge(min_edge[0], min_edge[1])
                print(f"Removed cycle edge: {min_edge[0]} -> {min_edge[1]}")
                
            except nx.NetworkXNoCycle:
                break
    
    def get_prerequisites(self, topic_id: int, recursive: bool = False) -> List[int]:
        """
        Get prerequisites for a given topic.
        
        Args:
            topic_id: ID of the topic
            recursive: If True, return all transitive prerequisites
            
        Returns:
            List of prerequisite topic IDs
        """
        if topic_id not in self.prerequisite_graph:
            return []
        
        if recursive:
            # Get all ancestors (transitive prerequisites)
            return list(nx.ancestors(self.prerequisite_graph, topic_id))
        else:
            # Get only direct prerequisites
            return list(self.prerequisite_graph.predecessors(topic_id))
    
    def get_learning_path(self, topic_id: int) -> List[int]:
        """
        Get a recommended learning path to reach a topic.
        Returns topics in topological order (prerequisites first).
        
        Args:
            topic_id: Target topic ID
            
        Returns:
            Ordered list of topic IDs to learn
        """
        if topic_id not in self.prerequisite_graph:
            return [topic_id]
        
        # Get subgraph of all prerequisites
        prerequisites = nx.ancestors(self.prerequisite_graph, topic_id)
        prerequisites.add(topic_id)
        subgraph = self.prerequisite_graph.subgraph(prerequisites)
        
        # Return in topological order
        try:
            return list(nx.topological_sort(subgraph))
        except nx.NetworkXError:
            # Fallback if graph has cycles
            return [topic_id]
    
    def get_next_topics(self, completed_topics: Set[int], max_recommendations: int = 5) -> List[Tuple[int, float]]:
        """
        Recommend next topics to learn based on completed topics.
        
        Args:
            completed_topics: Set of topic IDs already mastered
            max_recommendations: Maximum number of topics to recommend
            
        Returns:
            List of (topic_id, readiness_score) tuples sorted by readiness
        """
        candidates = []
        
        for topic_id in self.prerequisite_graph.nodes():
            if topic_id in completed_topics:
                continue
            
            # Get prerequisites for this topic
            prerequisites = set(self.get_prerequisites(topic_id, recursive=False))
            
            if not prerequisites:
                # No prerequisites, fully ready
                candidates.append((topic_id, 1.0))
            else:
                # Calculate readiness as percentage of prerequisites completed
                completed_prereqs = prerequisites.intersection(completed_topics)
                readiness = len(completed_prereqs) / len(prerequisites)
                
                # Only recommend if at least 50% prerequisites met
                if readiness >= 0.5:
                    candidates.append((topic_id, readiness))
        
        # Sort by readiness (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:max_recommendations]
    
    def visualize_graph(self, output_path: str = 'visualizations/prerequisite_graph.html', 
                       highlight_topics: Optional[List[int]] = None):
        """
        Create an interactive visualization of the prerequisite graph.
        
        Args:
            output_path: Path to save HTML visualization
            highlight_topics: List of topic IDs to highlight
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("pyvis not installed. Install with: pip install pyvis")
            return
        
        # Create network
        net = Network(height='750px', width='100%', directed=True, notebook=False)
        net.barnes_hut(gravity=-8000, spring_length=200)
        
        # Add nodes
        for node_id in self.prerequisite_graph.nodes():
            node_data = self.prerequisite_graph.nodes[node_id]
            
            # Color by difficulty
            difficulty = node_data.get('difficulty', 0.5)
            if difficulty < 0.33:
                color = '#90EE90'  # Light green (easy)
            elif difficulty < 0.67:
                color = '#FFD700'  # Gold (medium)
            else:
                color = '#FF6B6B'  # Red (hard)
            
            # Highlight if requested
            if highlight_topics and node_id in highlight_topics:
                color = '#4169E1'  # Blue
                border_width = 4
            else:
                border_width = 2
            
            title = f"{node_data['name']}\nDifficulty: {difficulty:.2f}\nResources: {node_data['num_resources']}"
            
            net.add_node(
                node_id,
                label=node_data['name'][:30],  # Truncate long names
                title=title,
                color=color,
                borderWidth=border_width
            )
        
        # Add edges
        for source, target, data in self.prerequisite_graph.edges(data=True):
            weight = data.get('weight', 0.5)
            net.add_edge(
                source,
                target,
                value=weight * 5,  # Scale for visibility
                title=f"Prerequisite strength: {weight:.3f}"
            )
        
        # Save
        net.show(output_path)
        print(f"Visualization saved to {output_path}")
    
    def export_graph(self, output_path: str = 'data/prerequisite_graph.json'):
        """
        Export the prerequisite graph to JSON format.
        
        Args:
            output_path: Path to save JSON file
        """
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Export nodes
        for node_id in self.prerequisite_graph.nodes():
            node_data = self.prerequisite_graph.nodes[node_id]
            graph_data['nodes'].append({
                'id': int(node_id),
                'name': node_data['name'],
                'difficulty': float(node_data.get('difficulty', 0.5)),
                'num_resources': int(node_data['num_resources'])
            })
        
        # Export edges
        for source, target, data in self.prerequisite_graph.edges(data=True):
            graph_data['edges'].append({
                'source': int(source),
                'target': int(target),
                'weight': float(data.get('weight', 0.5))
            })
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph exported to {output_path}")
    
    def print_statistics(self):
        """Print statistics about the prerequisite graph."""
        print("\n=== Prerequisite Graph Statistics ===")
        print(f"Number of topics: {self.prerequisite_graph.number_of_nodes()}")
        print(f"Number of prerequisite relationships: {self.prerequisite_graph.number_of_edges()}")
        
        if self.prerequisite_graph.number_of_nodes() > 0:
            # Calculate average prerequisites per topic
            prereq_counts = [len(list(self.prerequisite_graph.predecessors(n))) 
                           for n in self.prerequisite_graph.nodes()]
            avg_prereqs = np.mean(prereq_counts)
            max_prereqs = np.max(prereq_counts)
            
            print(f"Average prerequisites per topic: {avg_prereqs:.2f}")
            print(f"Maximum prerequisites for a topic: {max_prereqs}")
            
            # Find topics with no prerequisites (foundational topics)
            foundational = [n for n in self.prerequisite_graph.nodes() 
                          if len(list(self.prerequisite_graph.predecessors(n))) == 0]
            print(f"Foundational topics (no prerequisites): {len(foundational)}")
            
            # Find topics with no dependents (advanced/leaf topics)
            advanced = [n for n in self.prerequisite_graph.nodes()
                       if len(list(self.prerequisite_graph.successors(n))) == 0]
            print(f"Advanced topics (no dependents): {len(advanced)}")
            
            # Difficulty distribution
            difficulties = [self.topic_difficulties.get(n, 0.5) 
                          for n in self.prerequisite_graph.nodes()]
            print(f"Average topic difficulty: {np.mean(difficulties):.3f}")
            print(f"Difficulty range: [{np.min(difficulties):.3f}, {np.max(difficulties):.3f}]")
        
        print("=" * 40)


def main():
    """Example usage of the PrerequisiteGraphGenerator."""
    
    print("=== Prerequisite Graph Generator ===\n")
    
    # Initialize generator
    generator = PrerequisiteGraphGenerator(
        resources_path='data/resources_with_topics_integrated.csv',
        min_similarity=0.35,
        max_similarity=0.85,
        difficulty_weight=0.4
    )
    
    # Step 1: Load or generate embeddings
    print("Step 1: Loading embeddings...")
    
    
    # Step 2: Aggregate to topic level
    print("\nStep 2: Aggregating to topic level...")
    generator.aggregate_topic_embeddings()
    
    # Step 3: Estimate difficulties
    print("\nStep 3: Estimating topic difficulties...")
    generator.estimate_topic_difficulty()
    
    # Step 4: Build graph
    print("\nStep 4: Building prerequisite graph...")
    generator.build_prerequisite_graph(
        top_k_per_topic=5,
        min_score=0.4
    )
    
    # Print statistics
    generator.print_statistics()
    
    # Step 5: Export graph
    print("\nStep 5: Exporting graph...")
    generator.export_graph('data/prerequisite_graph.json')
    
    # Step 6: Visualize (optional)
    print("\nStep 6: Creating visualization...")
    try:
        generator.visualize_graph('visualizations/prerequisite_graph.html')
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # Example: Get learning path for a topic
    if len(generator.topic_info) > 0:
        sample_topic = list(generator.topic_info.keys())[0]
        print(f"\nExample learning path for topic {sample_topic}:")
        path = generator.get_learning_path(sample_topic)
        for topic_id in path:
            info = generator.topic_info.get(topic_id, {})
            print(f"  - {info.get('topic_name', f'Topic {topic_id}')}")
    
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
