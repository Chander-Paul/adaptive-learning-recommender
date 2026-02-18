"""
K-Prototypes Clustering model to learn possible student learning paths. 

Source Data: https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style/data
"""

from xml.sax.handler import all_features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes
from enum import Enum



import warnings
warnings.filterwarnings('ignore')

class FeatureType(Enum):
    continuous = 1
    categorical = 2

class DataType(Enum):
    binary = 1
    ordinal = 2
    integer = 3
    percentage = 4

class ClusterType(Enum):
    LearningStyle = 1
    Performance = 2
    Affective = 3




class KPrototypesLearningPathClusterer:
    """
    A class for clustering students using K-Prototypes algorithm
    which handles mixed data types (continuous and categorical).
    """
    
    def __init__(self, data_path='data/training_files/student_performance.csv', 
                 use_minmax_scaler=False):
        """
        Initialize the K-Prototypes clustering model.
        
        Parameters:
        -----------
        data_path : str
            Path to the student performance CSV file
        use_minmax_scaler : bool
            If True, use MinMaxScaler (scales to [0,1], keeps all positive)
            If False, use StandardScaler (z-score, mean=0, std=1, allows negatives)
        """
        self.data_path = data_path
        self.df = None
        self.df_features = None
        self.cluster_labels = None
        self.optimal_k = None
        self.model = None

        self.scaler = MinMaxScaler() if use_minmax_scaler else StandardScaler()
        self.scaler_type = 'MinMax [0,1]' if use_minmax_scaler else 'Standard (z-score)'
        

        # Define features
        self.features = {
            #learning style features
            'StudyHours': {FeatureType.continuous, DataType.integer, ClusterType.LearningStyle},
            'OnlineCourses': {FeatureType.continuous, DataType.integer, ClusterType.LearningStyle},
            'Discussions': {FeatureType.categorical, DataType.binary, ClusterType.LearningStyle  },
            'Resources': {FeatureType.categorical, DataType.binary, ClusterType.LearningStyle},
            'EduTech': {FeatureType.categorical, DataType.binary, ClusterType.LearningStyle},
            'Extracurricular': {FeatureType.categorical, DataType.binary, ClusterType.LearningStyle},
        
            #performance features
            'ExamScore': {FeatureType.continuous, DataType.percentage, ClusterType.Performance},
            'AssignmentCompletion': {FeatureType.continuous, DataType.percentage, ClusterType.Performance},
            'FinalGrade': {FeatureType.continuous, DataType.integer, ClusterType.Performance},
            #affective features
            'Attendance': {FeatureType.continuous, DataType.percentage, ClusterType.Affective},
            'Motivation': {FeatureType.categorical, DataType.ordinal, ClusterType.Affective},
            'StressLevel': {FeatureType.categorical, DataType.ordinal, ClusterType.Affective},
        
        }
        self.continuous_features = [f for f in self.features if FeatureType.continuous in self.features[f]]
        self.categorical_features = [f for f in self.features if FeatureType.categorical in self.features[f]]
        self.categorical_indices = []
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    
    
    def preprocess_data(self, exclude_features=['FinalGrade', 'Gender','ExamScore',]):
        """
        Preprocess data by separating continuous and categorical features.
        
        Parameters:
        -----------
        exclude_features : list
            Features to exclude from clustering
        """
        print("\n" + "="*60)
        print("Preprocess data for clustering")
        print("="*60)
        
        self.categorical_indices = list(range(len(self.continuous_features), 
                                             len(self.features)))
        
        # Load features from data
        self.df_features = self.df[list(self.features.keys())].copy()
        # Normalize continuous features
        print(f"\nNormalizing continuous features using {self.scaler_type}")
        continuous_data = self.df_features[self.continuous_features].values.astype(float)
        
        # Handle percentage features specially
        percentage_cols = []
        for col in self.continuous_features:
            if DataType.percentage in self.features[col]:
                idx = self.continuous_features.index(col)
                # Convert percentages to [0,1] range
                continuous_data[:, idx] /= 100.0
                percentage_cols.append(col)
        
        if percentage_cols:
            print(f"  Percentage features converted to [0,1]: {percentage_cols}")
        
        # Apply scaling
        # Note: For percentage data already in [0,1], StandardScaler may create negative values
        # Consider using MinMaxScaler (use_minmax_scaler=True) to keep values bounded
        continuous_normalized = self.scaler.fit_transform(continuous_data)
        
        # Replace continuous columns with normalized values
        for idx, col in enumerate(self.continuous_features):
            self.df_features[col] = continuous_normalized[:, idx]
        
        if isinstance(self.scaler, MinMaxScaler):
            print(f"  All continuous features scaled to [0, 1]")
        else:
            print(f"  All continuous features scaled to mean=0, std=1 (may include negative values)")
        
        # Ensure categorical features are integers
        for col in self.categorical_features:
            self.df_features[col] = self.df_features[col].astype(int)
        
        print(f"\nPrepared data shape: {self.df_features.shape}")
        print("\nSample of prepared data (continuous features normalized):")
        print(self.df_features.head())
        
        return self.df_features
    
    def find_optimal_clusters(self, max_k=8, n_init=5):
        """
        Find the optimal number of clusters using cost and silhouette methods.
        
        Parameters:
        -----------
        max_k : int
            Maximum number of clusters to test
        n_init : int
            Number of initializations for each k (higher = more stable but slower)
        """
        print("\n" + "="*60)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS (K-PROTOTYPES)")
        print("="*60)
        
        costs = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            print(f"\nTesting k={k}...")
            # Initialize and fit K-Prototypes
            kproto = KPrototypes(n_clusters=k, init='Cao', n_init=10, random_state=42, verbose=0, )
            
            labels = kproto.fit_predict(self.df_features.values, 
                                       categorical=self.categorical_indices)
            costs.append(kproto.cost_)
            # Calculate silhouette score using only continuous features
            # (silhouette_score requires numeric data)
            continuous_data = self.df_features[self.continuous_features].values
            silhouette = silhouette_score(continuous_data, labels)
            silhouette_scores.append(silhouette)
            print(f"  Cost: {kproto.cost_:.2f}, Silhouette: {silhouette:.3f}")
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # Cost plot (Elbow method)
        axes[0].plot(k_range, costs, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Cost', fontsize=12)
        axes[0].set_title('K-Prototypes Cost (Elbow Method)', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        # Silhouette plot
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/kprototypes_optimization.png', dpi=300, bbox_inches='tight')
        print("\n✓ Cluster optimization plot saved to 'visualizations/kprototypes_optimization.png'")
        
        # Determine optimal k
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n✓ Optimal number of clusters based on Silhouette Score: {self.optimal_k}")
        
        # Calculate elbow point using the elbow method
        # Find the point with maximum distance from the line connecting first and last points
        costs_array = np.array(costs)
        n_points = len(costs_array)
        all_coords = np.vstack((range(len(costs_array)), costs_array)).T
        first_point = all_coords[0]
        last_point = all_coords[-1]
        
        line_vec = last_point - first_point
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        
        vec_from_first = all_coords - first_point
        scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
        dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
        
        elbow_k = k_range[np.argmax(dist_to_line)]
        print(f"✓ Elbow point detected at k={elbow_k}")
        
        return self.optimal_k, costs, silhouette_scores
    
    def perform_kprototypes_clustering(self, n_clusters=None, n_init=5):
        """
        Perform K-Prototypes clustering on the student data.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (uses optimal_k if None)
        n_init : int
            Number of initializations
        """
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 4
        
        print("\n" + "="*60)
        print(f"K-PROTOTYPES CLUSTERING (k={n_clusters})")
        print("="*60)
        
        # Initialize and fit K-Prototypes
        self.model = KPrototypes(n_clusters=n_clusters, 
                                init='Huang', 
                                n_init=n_init,
                                random_state=42,
                                verbose=1)
        
        self.cluster_labels = self.model.fit_predict(self.df_features.values, 
                                                     categorical=self.categorical_indices)
        
        # Add cluster labels to original dataframe
        self.df['Cluster'] = self.cluster_labels
        
        # Calculate metrics using continuous features
        continuous_data = self.df_features[self.continuous_features].values
        silhouette = silhouette_score(continuous_data, self.cluster_labels)
        
        print(f"\n✓ Clustering Metrics:")
        print(f"  Cost: {self.model.cost_:.2f}")
        print(f"  Silhouette Score (continuous features): {silhouette:.3f}")
        
        print(f"\n✓ Cluster Distribution:")
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} students ({count/len(self.df)*100:.1f}%)")
        
        return self.model
    
    def analyze_clusters(self):
        """Analyze and profile each cluster."""
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS & PROFILING")
        print("="*60)
        
        cluster_profiles = []
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            
            print(f"\n{'='*50}")
            print(f"CLUSTER {cluster_id}: {len(cluster_data)} students ({len(cluster_data)/len(self.df)*100:.1f}%)")
            print(f"{'='*50}")
            
            profile = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
                'Percentage': f"{len(cluster_data)/len(self.df)*100:.1f}%"
            }
            
            # Analyze continuous features
            print("\nContinuous Features:")
            for feature in self.continuous_features:
                if feature in cluster_data.columns:
                    mean_val = cluster_data[feature].mean()
                    std_val = cluster_data[feature].std()
                    profile[f'{feature}_mean'] = f"{mean_val:.2f}"
                    profile[f'{feature}_std'] = f"{std_val:.2f}"
                    print(f"  {feature}: {mean_val:.2f} ± {std_val:.2f}")
            
            # Analyze categorical features
            print("\nCategorical Features:")
            for feature in self.categorical_features:
                if feature in cluster_data.columns:
                    mode_val = cluster_data[feature].mode()[0]
                    value_counts = cluster_data[feature].value_counts()
                    profile[f'{feature}_mode'] = mode_val
                    
                    # For binary features, show percentage
                    if len(cluster_data[feature].unique()) <= 2:
                        pct = cluster_data[feature].mean() * 100
                        profile[f'{feature}_pct'] = f"{pct:.1f}%"
                        print(f"  {feature}: {pct:.1f}% (mode={mode_val})")
                    else:
                        print(f"  {feature}: mode={mode_val}, distribution={dict(value_counts)}")
            
            # Target variable analysis
            if 'FinalGrade' in cluster_data.columns:
                avg_grade = cluster_data['FinalGrade'].mean()
                profile['FinalGrade_mean'] = f"{avg_grade:.2f}"
                print(f"\nFinal Grade (Average): {avg_grade:.2f}")
            
            cluster_profiles.append(profile)
        
        # Create profile dataframe
        profile_df = pd.DataFrame(cluster_profiles)
        
        print("\n" + "="*60)
        print("CLUSTER PROFILES SUMMARY")
        print("="*60)
        print(profile_df.to_string())
        
        # Save profiles
        profile_df.to_csv('visualizations/kprototypes_cluster_profiles.csv', index=False)
        print("\n✓ Cluster profiles saved to 'visualizations/kprototypes_cluster_profiles.csv'")
        
        return profile_df
    
    def visualize_clusters_pca(self):
        """Visualize clusters using PCA on continuous features."""
        print("\n" + "="*60)
        print("PCA VISUALIZATION (Continuous Features)")
        print("="*60)
        
        # Apply PCA on continuous features only
        continuous_data = self.df_features[self.continuous_features].values
        
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(continuous_data)
        
        print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1],
                            c=self.cluster_labels, cmap='viridis', 
                            alpha=0.6, edgecolors='k', linewidth=0.5)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title('K-Prototypes Clusters Visualization (PCA on Continuous Features)', fontsize=14)
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/kprototypes_clusters_pca.png', dpi=300, bbox_inches='tight')
        print("\n✓ PCA visualization saved to 'visualizations/kprototypes_clusters_pca.png'")
        
        return pca_features
    
    def visualize_cluster_distributions(self):
        """Create detailed visualizations of cluster characteristics."""
        print("\n" + "="*60)
        print("CREATING CLUSTER DISTRIBUTION VISUALIZATIONS")
        print("="*60)
        
        # Continuous features
        n_continuous = len(self.continuous_features)
        n_cols = 3
        n_rows = (n_continuous + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.continuous_features):
            if feature in self.df.columns:
                for cluster_id in sorted(self.df['Cluster'].unique()):
                    cluster_data = self.df[self.df['Cluster'] == cluster_id]
                    axes[idx].hist(cluster_data[feature], alpha=0.5, 
                                  label=f'Cluster {cluster_id}', bins=20)
                
                axes[idx].set_xlabel(feature, fontsize=10)
                axes[idx].set_ylabel('Frequency', fontsize=10)
                axes[idx].set_title(f'{feature} by Cluster', fontsize=11)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        for idx in range(n_continuous, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/kprototypes_continuous_distributions.png', dpi=300, bbox_inches='tight')
        print("\n✓ Continuous distributions saved")
        
        # Categorical features visualization
        n_categorical = len(self.categorical_features)
        n_rows_cat = (n_categorical + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows_cat, n_cols, figsize=(18, n_rows_cat * 4))
        axes = axes.flatten() if n_categorical > 1 else [axes]
        
        for idx, feature in enumerate(self.categorical_features):
            if feature in self.df.columns:
                cluster_feature_counts = self.df.groupby(['Cluster', feature]).size().unstack(fill_value=0)
                cluster_feature_counts.plot(kind='bar', ax=axes[idx], stacked=False)
                axes[idx].set_title(f'{feature} Distribution by Cluster', fontsize=11)
                axes[idx].set_xlabel('Cluster', fontsize=10)
                axes[idx].set_ylabel('Count', fontsize=10)
                axes[idx].legend(title=feature)
                axes[idx].grid(True, alpha=0.3, axis='y')
        
        for idx in range(n_categorical, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/kprototypes_categorical_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Categorical distributions saved")
    
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
            motivation_mode = cluster_data['Motivation'].mode()[0]
            stress_mode = cluster_data['StressLevel'].mode()[0]
            
            # Characterize the learning path
            if avg_exam_score >= 70 and avg_final_grade <= 1:
                path_type = "HIGH ACHIEVERS"
                description = "Students with excellent performance and high engagement"
                recommendations = [
                    "Provide advanced/challenging materials",
                    "Offer leadership and mentorship opportunities",
                    "Encourage peer tutoring and collaboration"
                ]
            elif avg_study_hours >= 25 and avg_exam_score < 70:
                path_type = "STRUGGLING HARD WORKERS"
                description = "Students putting in effort but facing difficulties"
                recommendations = [
                    "Provide personalized tutoring and support",
                    "Review study strategies and learning techniques",
                    "Offer additional practice materials and resources",
                    "Consider learning style adaptations"
                ]
            elif avg_attendance < 70:
                path_type = "DISENGAGED LEARNERS"
                description = "Students with low attendance and engagement"
                recommendations = [
                    "Increase motivation through interactive content",
                    "Provide flexible learning options",
                    "Address underlying issues (stress, resources, motivation)",
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
            
            learning_paths[cluster_id] = {
                'Type': path_type,
                'Description': description,
                'Size': int(len(cluster_data)),
                'Percentage': f"{len(cluster_data)/len(self.df)*100:.1f}%",
                'Avg_ExamScore': f"{avg_exam_score:.2f}",
                'Avg_StudyHours': f"{avg_study_hours:.2f}",
                'Avg_Attendance': f"{avg_attendance:.2f}%",
                'Avg_Assignment': f"{avg_assignment:.2f}%",
                'Avg_FinalGrade': f"{avg_final_grade:.2f}",
                'Avg_OnlineCourses': f"{online_courses:.2f}",
                'Motivation_Mode': int(motivation_mode),
                'StressLevel_Mode': int(stress_mode),
                'Recommendations': recommendations
            }
            
            print(f"\nCluster {cluster_id}: {path_type}")
            print(f"  Description: {description}")
            print(f"  Students: {len(cluster_data)} ({len(cluster_data)/len(self.df)*100:.1f}%)")
            print(f"  Avg Exam Score: {avg_exam_score:.2f}")
            print(f"  Avg Study Hours: {avg_study_hours:.2f}")
            print(f"  Avg Attendance: {avg_attendance:.2f}%")
            print(f"  Avg Final Grade: {avg_final_grade:.2f}")
            print(f"  Motivation (mode): {motivation_mode}, Stress (mode): {stress_mode}")
            print("  Recommendations:")
            for rec in recommendations:
                print(f"    • {rec}")
        
        # Save learning paths
        import json
        # Convert numpy types to Python native types
        learning_paths_serializable = {}
        for k, v in learning_paths.items():
            learning_paths_serializable[int(k)] = v
        
        with open('visualizations/kprototypes_learning_paths.json', 'w') as f:
            json.dump(learning_paths_serializable, f, indent=2)
        print("\n✓ Learning paths saved to 'visualizations/kprototypes_learning_paths.json'")
        
        return learning_paths
    
    def predict_cluster(self, student_features):
        """
        Predict the cluster/learning path for a new student.
        
        Parameters:
        -----------
        student_features : dict or pd.DataFrame
            Student features matching the training data structure
        """
        if self.model is None:
            raise ValueError("Model not trained. Call perform_kprototypes_clustering first.")
        
        if isinstance(student_features, dict):
            student_df = pd.DataFrame([student_features])
        else:
            student_df = student_features
        
        # Ensure features are in the correct order
        student_continuous_features = [f for f in self.continuous_features if f in student_df.columns]
        student_categorical_features = [f for f in self.categorical_features if f in student_df.columns]
        combined_features = student_continuous_features + student_categorical_features
        all_features = [f for f in combined_features if f in student_df.columns]
        student_data = student_df[all_features].copy()
        # Normalize continuous features using the fitted scaler
        continuous_data = student_data[student_continuous_features].values.astype(float)
        
        # Apply same percentage conversion as during training
        for col in student_continuous_features:
            if DataType.percentage in self.features[col]:
                idx = student_continuous_features.index(col)
                continuous_data[:, idx] /= 100.0
        if continuous_data.shape[1] > 0:
            continuous_normalized = self.scaler.transform(continuous_data)
        
            # Replace continuous columns with normalized values
            for idx, col in enumerate(student_continuous_features):
                student_data[col] = continuous_normalized[:, idx]
        
        # Predict cluster
        predicted_cluster = self.model.predict(student_data.values, 
                                              categorical=self.categorical_indices)[0]
        
        return predicted_cluster
    
    def save_model(self, filepath='models/kprototypes_learning_path_model.pkl'):
        """Save the trained model."""
        import pickle
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'continuous_features': self.continuous_features,
            'categorical_features': self.categorical_features,
            'categorical_indices': self.categorical_indices,
            'optimal_k': self.optimal_k,
            'cluster_labels': self.cluster_labels
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to '{filepath}'")
    
    def run_full_analysis(self):
        """Run the complete clustering analysis pipeline."""
        print("\n" + "="*60)
        print("K-PROTOTYPES LEARNING PATH CLUSTERING ANALYSIS")
        print("="*60)
        

        self.load_data()
        self.preprocess_data()
        self.find_optimal_clusters(max_k=8)
        self.perform_kprototypes_clustering()
        self.analyze_clusters()
        self.visualize_clusters_pca()
        self.visualize_cluster_distributions()
        learning_paths = self.identify_learning_paths()
        self.save_model()

        
        return learning_paths

    def load_model(self, filepath='models/kprototypes_learning_path_model.pkl'):
        """Load a saved K-Prototypes model."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.continuous_features = model_data['continuous_features']
        self.categorical_features = model_data['categorical_features']
        self.categorical_indices = model_data['categorical_indices']
        self.optimal_k = model_data['optimal_k']
        self.cluster_labels = model_data['cluster_labels']
        
        return model_data
