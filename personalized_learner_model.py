"""
Models for clustering and predicting learning paths and recommended actions. 


https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style/data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings('ignore')


class LearningPathClusterer:
    """
    A class for clustering students into learning paths based on their performance
    and behavioral characteristics.
    """
    
    def __init__(self, data_path='data/training_files/student_performance.csv'):
        """ 
        data_path : str
        """
        self.data_path = data_path
        self.features = ["StudyHours", "Attendance", "Resources",
                              "Motivation","Internet", "Age","OnlineCourses","Discussions",
                              "AssignmentCompletion", "EduTech", "LearningStyle",
                              ]
        self.df = None
        self.model = None
        self.processed_data = None
        self.scale = True
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.optimal_k = None
        self.pca_features = None
        
    def load_data(self):
        """Load and preprocess the student performance data."""
        print("Loading student performance data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} student records")
        print(f"Features: {self.features}")
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*60)
        print("DATA EXPLORATION")
        print("="*60)
        
        print("\nDataset Shape:", self.df.shape)
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nData Info:")
        print(self.df.info())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nFeature Correlations with FinalGrade:")
        correlations = self.df.corr()['FinalGrade'].sort_values(ascending=False)
        print(correlations)
        
        return correlations
    
    def preprocess_data(self):
        """        
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Select features for clustering

        
        print(f"\nClustering features: {self.features}")
        if self.scale:
            print("Scaling features using StandardScaler...")
        # Scale the features
            print(self.df[self.features].head(10))
            self.processed_data = self.scaler.fit(self.df[self.features])
            self.processed_data = self.scaler.transform(self.df[self.features])
            print(f"Scaled data shape: {self.processed_data.shape}")
            
        else:
            print(self.df[self.features].values[:10])
            self.processed_data = self.df[self.features].values
            print(f"Data shape without scaling: {self.processed_data.shape}")

        
        
    
    def find_optimal_clusters(self, max_k=10, method='elbow'):
        """
        Find the optimal number of clusters using various methods.
        
        Parameters:
        -----------
        max_k : int
            Maximum number of clusters to test
        method : str
            Method to use ('elbow', 'silhouette', or 'both')
        """
        print("\n" + "="*60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*60)
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.processed_data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.processed_data, labels))
            davies_bouldin_scores.append(davies_bouldin_score(self.processed_data, labels))
            calinski_harabasz_scores.append(calinski_harabasz_score(self.processed_data, labels))
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, "
                  f"Silhouette={silhouette_scores[-1]:.3f}, "
                  f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}, "
                  f"Calinski-Harabasz={calinski_harabasz_scores[-1]:.2f}")
        
        # Plot elbow curve
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].plot(k_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method - Inertia')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(k_range, silhouette_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score (Higher is Better)')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(k_range, davies_bouldin_scores, 'go-')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Davies-Bouldin Score')
        axes[1, 0].set_title('Davies-Bouldin Score (Lower is Better)')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(k_range, calinski_harabasz_scores, 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Calinski-Harabasz Score')
        axes[1, 1].set_title('Calinski-Harabasz Score (Higher is Better)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/cluster_optimization.png', dpi=300, bbox_inches='tight')
        print("\nCluster optimization plot saved to 'visualizations/cluster_optimization.png'")
        
        # Determine optimal k based on silhouette score
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters based on Silhouette Score: {self.optimal_k}")
        
        return self.optimal_k, silhouette_scores, inertias
    
    def perform_kmeans_clustering(self, n_clusters=None):
        """
        Perform K-Means clustering on the student data.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (uses optimal_k if None)
        """
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 4
        
        print("\n" + "="*60)
        print(f"K-MEANS CLUSTERING (k={n_clusters})")
        print("="*60)
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.model.fit_predict(self.processed_data)
        
        # Add cluster labels to original dataframe
        self.df['Cluster'] = self.cluster_labels
        
        # Calculate metrics
        silhouette = silhouette_score(self.processed_data, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.processed_data, self.cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.processed_data, self.cluster_labels)
        
        print(f"\nClustering Metrics:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        
        print(f"\nCluster Distribution:")
        print(self.df['Cluster'].value_counts().sort_index())
        
        return self.model
    
    def perform_hierarchical_clustering(self, n_clusters=None):
        """
        Perform Hierarchical clustering on the student data.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (uses optimal_k if None)
        """
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 4
        
        print("\n" + "="*60)
        print(f"HIERARCHICAL CLUSTERING (k={n_clusters})")
        print("="*60)
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.cluster_labels = hierarchical.fit_predict(self.processed_data)
        
        # Add cluster labels to original dataframe
        self.df['Cluster_Hierarchical'] = self.cluster_labels
        
        # Calculate metrics
        silhouette = silhouette_score(self.processed_data, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.processed_data, self.cluster_labels)
        
        print(f"\nClustering Metrics:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
        
        # Create dendrogram
        plt.figure(figsize=(15, 6))
        linkage_matrix = linkage(self.processed_data[:1000], method='ward')  # Sample for visualization
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram (Sample of 1000 students)')
        plt.xlabel('Student Index')
        plt.ylabel('Distance')
        plt.savefig('visualizations/dendrogram.png', dpi=300, bbox_inches='tight')
        print("\nDendrogram saved to 'visualizations/dendrogram.png'")
        
        return hierarchical
    
    def analyze_clusters(self):
        """Analyze and profile each cluster."""
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS & PROFILING")
        print("="*60)
        
        cluster_profiles = []
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            
            print(f"\n{'='*40}")
            print(f"CLUSTER {cluster_id}: {len(cluster_data)} students")
            print(f"{'='*40}")
            
            profile = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
                'Percentage': f"{len(cluster_data)/len(self.df)*100:.1f}%"
            }
            
            # Numerical features
        
            
            for feature in self.features:
                if feature in cluster_data.columns:
                    mean_val = cluster_data[feature].mean()
                    profile[f'{feature}_mean'] = mean_val
                    print(f"{feature}: {mean_val:.2f}")
            
            cluster_profiles.append(profile)
        
        # Create profile dataframe
        profile_df = pd.DataFrame(cluster_profiles)
        
        print("\n" + "="*60)
        print("CLUSTER PROFILES SUMMARY")
        print("="*60)
        print(profile_df.to_string())
        
        # Save profiles
        profile_df.to_csv('visualizations/cluster_profiles.csv', index=False)
        print("\nCluster profiles saved to 'visualizations/cluster_profiles.csv'")
        
        return profile_df
    
    def visualize_clusters_pca(self):
        """Visualize clusters using PCA for dimensionality reduction."""
        print("\n" + "="*60)
        print("PCA VISUALIZATION")
        print("="*60)
        
        # Apply PCA
        pca = PCA(n_components=2)
        self.pca_features = pca.fit_transform(self.processed_data)
        
        print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.pca_features[:, 0], self.pca_features[:, 1],
                            c=self.cluster_labels, cmap='viridis', alpha=0.6)
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Student Clusters Visualization (PCA)')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/clusters_pca.png', dpi=300, bbox_inches='tight')
        print("\nPCA visualization saved to 'visualizations/clusters_pca.png'")
        
        return self.pca_features
    
    def visualize_cluster_distributions(self):
        """Create detailed visualizations of cluster characteristics."""
        print("\n" + "="*60)
        print("CREATING CLUSTER DISTRIBUTION VISUALIZATIONS")
        print("="*60)
        
        # Key features to visualize
        key_features = self.features
        
        n_features = len(key_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(key_features):
            if feature in self.df.columns:
                for cluster_id in sorted(self.df['Cluster'].unique()):
                    cluster_data = self.df[self.df['Cluster'] == cluster_id]
                    axes[idx].hist(cluster_data[feature], alpha=0.5, 
                                  label=f'Cluster {cluster_id}', bins=20)
                
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Frequency')
                axes[idx].set_title(f'Distribution of {feature} by Cluster')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/cluster_distributions.png', dpi=300, bbox_inches='tight')
        print("\nCluster distributions saved to 'visualizations/cluster_distributions.png'")
        
        # Box plots for key metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        key_metrics = ['ExamScore', 'AssignmentCompletion', 'StudyHours', 'Attendance']
        for idx, metric in enumerate(key_metrics):
            row, col = idx // 2, idx % 2
            self.df.boxplot(column=metric, by='Cluster', ax=axes[row, col])
            axes[row, col].set_title(f'{metric} by Cluster')
            axes[row, col].set_xlabel('Cluster')
            axes[row, col].set_ylabel(metric)
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('visualizations/cluster_boxplots.png', dpi=300, bbox_inches='tight')
        print("Cluster boxplots saved to 'visualizations/cluster_boxplots.png'")
    
    def identify_learning_paths(self):
        """
        Identify and describe learning paths based on cluster characteristics.
        """
        print("\n" + "="*60)
        print("LEARNING PATH IDENTIFICATION")
        print("="*60)
        
        learning_paths = {}
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            
            # Calculate key metrics
            avg_exam_score = cluster_data['ExamScore'].mean()
            avg_study_hours = cluster_data['StudyHours'].mean()
            avg_attendance = cluster_data['Attendance'].mean()
            avg_assignment = cluster_data['AssignmentCompletion'].mean()
            avg_final_grade = cluster_data['FinalGrade'].mean()
            online_courses = cluster_data['OnlineCourses'].mean()
            
            # Characterize the learning path
            if avg_exam_score >= 70 and avg_final_grade <= 1:
                path_type = "HIGH ACHIEVERS"
                description = "Students with excellent performance and high engagement"
                recommendations = [
                    "Provide advanced/challenging materials",
                    "Offer leadership and mentorship opportunities",
                    "Encourage peer tutoring"
                ]
            elif avg_study_hours >= 25 and avg_exam_score < 70:
                path_type = "STRUGGLING HARD WORKERS"
                description = "Students putting in effort but facing difficulties"
                recommendations = [
                    "Provide personalized tutoring",
                    "Review study strategies and techniques",
                    "Offer additional practice materials",
                    "Consider learning style adaptations"
                ]
            elif avg_attendance < 70:
                path_type = "DISENGAGED LEARNERS"
                description = "Students with low attendance and engagement"
                recommendations = [
                    "Increase motivation through interactive content",
                    "Provide flexible learning options",
                    "Address underlying issues (stress, resources)",
                    "Implement early intervention programs"
                ]
            elif avg_final_grade <= 1.5 and avg_attendance >= 85:
                path_type = "CONSISTENT PERFORMERS"
                description = "Reliable students with steady performance"
                recommendations = [
                    "Maintain current learning approach",
                    "Provide opportunities for skill enhancement",
                    "Encourage exploration of advanced topics"
                ]
            else:
                path_type = "MODERATE PERFORMERS"
                description = "Students with average performance and potential for growth"
                recommendations = [
                    "Identify specific areas for improvement",
                    "Provide targeted resources and support",
                    "Encourage consistent study habits",
                    "Monitor progress regularly"
                ]
            
            learning_paths[str(cluster_id)] = {
                'Type': path_type,
                'Description': description,
                'Size': len(cluster_data),
                'Percentage': f"{len(cluster_data)/len(self.df)*100:.1f}%",
                'Avg_ExamScore': f"{avg_exam_score:.2f}",
                'Avg_StudyHours': f"{avg_study_hours:.2f}",
                'Avg_Attendance': f"{avg_attendance:.2f}%",
                'Avg_Assignment': f"{avg_assignment:.2f}%",
                'Avg_FinalGrade': f"{avg_final_grade:.2f}",
                'Avg_OnlineCourses': f"{online_courses:.2f}",
                'Recommendations': recommendations
            }
            
            print(f"\nCluster {cluster_id}: {path_type}")
            print(f"Description: {description}")
            print(f"Students: {len(cluster_data)} ({len(cluster_data)/len(self.df)*100:.1f}%)")
            print(f"Avg Exam Score: {avg_exam_score:.2f}")
            print(f"Avg Study Hours: {avg_study_hours:.2f}")
            print(f"Avg Attendance: {avg_attendance:.2f}%")
            print(f"Avg Final Grade: {avg_final_grade:.2f}")
            print("Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        # Save learning paths
        import json
        with open('visualizations/learning_paths.json', 'w') as f:
            
            
            json.dump(learning_paths, f, indent=2)
        print("\nLearning paths saved to 'visualizations/learning_paths.json'")
        
        return learning_paths
    
    def predict_cluster(self, student_features):
        """
        Predict the cluster/learning path for a new student.
        
        Parameters:
        -----------
        student_features : dict or pd.DataFrame
            Student features matching the training data structure
        """
        if isinstance(student_features, dict):
            student_df = pd.DataFrame([student_features])
        else:
            student_df = student_features
        
        # Ensure features are in the same order as training
        if self.scale:
            if not hasattr(self.scaler, "mean_"):
                raise ValueError(
                    "Scaler is not fitted. Run preprocess_data() with scale=True or load a trained model."
                )
            student_features_scaled = self.scaler.transform(student_df[self.features])
        else:
            student_features_scaled = student_df[self.features].values
        
        # Predict cluster

        predicted_cluster = self.model.predict(student_features_scaled)[0]
        
 
        
        return predicted_cluster
    
    def save_model(self, filepath='models/learning_path_model.pkl'):
        """Save the trained model and scaler."""
        import pickle
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'scale': self.scale,
            'optimal_k': self.optimal_k,
            'features': self.features,
            'cluster_labels': self.cluster_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to '{filepath}'")
    
    def run_full_analysis(self):
        """Run the complete clustering analysis pipeline."""
        print("\n" + "="*60)
        print("LEARNING PATH CLUSTERING ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Preprocess
        self.preprocess_data()
        
        # Find optimal clusters
        self.find_optimal_clusters(max_k=8)
        
        # Perform clustering
        self.perform_kmeans_clustering()
        
        # Analyze clusters
        self.analyze_clusters()
        
        # Visualize
        self.visualize_clusters_pca()
        self.visualize_cluster_distributions()
        
        # Identify learning paths
        learning_paths = self.identify_learning_paths()
        
        # Save model
        self.save_model()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - visualizations/cluster_optimization.png")
        print("  - visualizations/clusters_pca.png")
        print("  - visualizations/cluster_distributions.png")
        print("  - visualizations/cluster_boxplots.png")
        print("  - visualizations/cluster_profiles.csv")
        print("  - visualizations/learning_paths.json")
        print("  - models/learning_path_model.pkl")
        
        return learning_paths

    def load_model(self, filepath='models/learning_path_model.pkl'):
        """Load a saved learning path model."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        print(f"\n✓ Model loaded from '{filepath}'")
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.scale = model_data['scale']
        self.optimal_k = model_data['optimal_k']
        self.features = model_data['features']
        self.cluster_labels = model_data['cluster_labels']
        
        print(f"\n✓ Model loaded from '{filepath}'")
        return model_data

def main():
    """Main execution function."""
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize and run the clustering model
    clusterer = LearningPathClusterer(data_path='data/training_files/student_performance.csv')
    learning_paths = clusterer.run_full_analysis()
    
    # Example: Predict cluster for a new student
    print("\n" + "="*60)
    print("EXAMPLE: PREDICT LEARNING PATH FOR NEW STUDENT")
    print("="*60)
    
    new_student = {
        'StudyHours': 25,
        'Attendance': 90,
        'Resources': 1,
        'Extracurricular': 1,
        'Motivation': 2,
        'Internet': 1,
        'Gender': 1,
        'Age': 22,
        'LearningStyle': 2,
        'OnlineCourses': 10,
        'Discussions': 1,
        'AssignmentCompletion': 85,
        'ExamScore': 75,
        'EduTech': 1,
        'StressLevel': 1
    }
    
    predicted_cluster = clusterer.predict_cluster(new_student)
    print(f"\nNew student predicted to be in Cluster {predicted_cluster}")
    print(f"Learning Path: {learning_paths[str(predicted_cluster)]['Type']}")
    print(f"Description: {learning_paths[str(predicted_cluster)]['Description']}")


if __name__ == "__main__":
    main()
