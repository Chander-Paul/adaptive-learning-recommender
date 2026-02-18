"""
Multi-View Clustering K-Prototypes for Student Learning Paths
Due to lower silhoutte scores in single-view clustering, this module implements
a multi-view clustering approach that segments students across three dimensions.

The idea being a student can have a distinct learning styles however their performance
and affective levels may vary independently. By clustering each dimension separately we're
hoping to identify these sepearate clusters and apply them to each student. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kprototypes_learner_model import KPrototypesLearningPathClusterer, FeatureType, DataType
from sklearn import cluster as kmeans
import json
import os

from personalized_learner_model import LearningPathClusterer


class MultiViewLearningPathClusterer:
    """
    Multi-view clustering that separates students into different dimensions:
    1. Learning Style/Behavior
    2. Performance 
    3. Affective
    """
    
    def __init__(self, data_path='data/training_files/student_performance.csv',
                 use_minmax_scaler=True):
        """
        Initialize multi-view clustering.
        
        Parameters:
        -----------
        data_path : str
            Path to student performance data
        use_minmax_scaler : bool
            Whether to use MinMaxScaler for normalization
        """
        self.data_path = data_path
        self.use_minmax_scaler = use_minmax_scaler
        self.df = None
        
        # Define feature sets for each view
        self.learning_style_features = {
            'StudyHours': {FeatureType.continuous, DataType.integer},
            'OnlineCourses': {FeatureType.continuous, DataType.integer},
            'Discussions': {FeatureType.categorical, DataType.binary},
            'Resources': {FeatureType.categorical, DataType.binary},
            'EduTech': {FeatureType.categorical, DataType.binary},
            'Extracurricular': {FeatureType.categorical, DataType.binary},
            #'LearningStyle': {FeatureType.categorical, DataType.nominal}
        }
        
        self.learning_performance_features = {
            'ExamScore': {FeatureType.continuous, DataType.percentage},
            'AssignmentCompletion': {FeatureType.continuous, DataType.percentage},
        }
        
        self.affective_features = {
            'Attendance': {FeatureType.continuous, DataType.percentage},
            'Motivation': {FeatureType.categorical, DataType.ordinal},
            'StressLevel': {FeatureType.categorical, DataType.ordinal},
        }
        
        # Clusterers for each view
        self.style_clusterer = KPrototypesLearningPathClusterer()
        self.performance_clusterer =  LearningPathClusterer()
        self.affective_clusterer = KPrototypesLearningPathClusterer()
        
        # Results
        self.style_labels = None
        self.performance_labels = None
        self.affective_labels = None
        
    def load_data(self):
        """Load the student data."""
        print("="*70)
        print("MULTI-VIEW CLUSTERING: Loading Data")
        print("="*70)
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} student records")
        return self.df
    

   


    def cluster_learning_styles(self, n_clusters=None):
        """
        Cluster students based on learning behaviors and preferences.
        """
        print("\n" + "="*70)
        print("VIEW 1: LEARNING STYLE CLUSTERING")
        print("="*70)
        
        self.style_clusterer = KPrototypesLearningPathClusterer(
            data_path=self.data_path,
            use_minmax_scaler=self.use_minmax_scaler
        )
        
  
        self.style_clusterer.features = self.learning_style_features
        self.style_clusterer.continuous_features = [f for f in self.learning_style_features 
                                                    if FeatureType.continuous in self.learning_style_features[f]]
        self.style_clusterer.categorical_features = [f for f in self.learning_style_features 
                                                     if FeatureType.categorical in self.learning_style_features[f]]
        
        # Run clustering
        self.style_clusterer.load_data()
        self.style_clusterer.preprocess_data(exclude_features=['FinalGrade', 'Gender'])
        
        if n_clusters is None:
            self.style_clusterer.find_optimal_clusters(max_k=6)
            n_clusters = self.style_clusterer.optimal_k
        
        self.style_clusterer.perform_kprototypes_clustering(n_clusters=n_clusters, n_init=5)
        profile_df = self.style_clusterer.analyze_clusters()
        
        self.style_labels = self.style_clusterer.cluster_labels
        self.df['LearningStyle_Cluster'] = self.style_labels
        
        # Visualize learning style clusters
        self._visualize_cluster_view(
            'LearningStyle',
            self.style_clusterer.df_features[self.style_clusterer.continuous_features].values,
            self.style_labels,
            n_clusters
        )
        
        # Interpret learning styles
        style_interpretations = self.interpret_learning_styles(profile_df)
        
        return self.style_labels, style_interpretations
    

    ## Change to one function that switches between kmeans and kprototpes based on the availablility
    ## of categorical features
    def cluster_performance(self, n_clusters=None):
        """
        Cluster students based on learning behaviors and preferences.
        """
        print("\n" + "="*70)
        print("VIEW 2: LEARNING PERFORMANCE CLUSTERING")
        print("="*70)
        from personalized_learner_model import LearningPathClusterer
        self.performance_clusterer = LearningPathClusterer(
            data_path=self.data_path,

        )
        
        self.performance_clusterer = LearningPathClusterer(data_path='data/training_files/student_performance.csv')
        
        #conver features to list of feature names
        continuous_features = [f for f in self.learning_performance_features 
                                if FeatureType.continuous in self.learning_performance_features[f]] 
        categorical_features = [f for f in self.learning_performance_features 
                                if FeatureType.categorical in self.learning_performance_features[f]]    
        
        self.performance_clusterer.features = continuous_features           
        
        # Run clustering       

        analysis = self.performance_clusterer.run_full_analysis()
        profile_df = self.performance_clusterer.analyze_clusters()
        
        self.performance_labels = self.performance_clusterer.cluster_labels
        self.df['LearningPerformance_Cluster'] = self.performance_labels
        
        # Visualize learning performance clusters
        self._visualize_cluster_view(
            'LearningPerformance',
            self.performance_clusterer.df[list(self.performance_clusterer.features)].copy().values,
            self.performance_labels,
            n_clusters
        )
        
        # Interpret learning performances
        performance_interpretations = self.interpret_performance(profile_df)
        self.performance_labels = self.performance_clusterer.cluster_labels
        self.df['Performance_Cluster'] = self.performance_labels # Save the trained model for future use
        return self.performance_labels, performance_interpretations
    
    def cluster_affective(self, n_clusters=3):
        """
        Cluster students based on affective levels.
        """
        print("\n" + "="*70)
        print("VIEW 3: ENGAGEMENT CLUSTERING")
        print("="*70)
        
        self.affective_clusterer = KPrototypesLearningPathClusterer(
            data_path=self.data_path,
            use_minmax_scaler=self.use_minmax_scaler
        )
        
        # Override features
        self.affective_clusterer.features = self.affective_features
        self.affective_clusterer.continuous_features = [f for f in self.affective_features 
                                                         if FeatureType.continuous in self.affective_features[f]]
        self.affective_clusterer.categorical_features = [f for f in self.affective_features 
                                                          if FeatureType.categorical in self.affective_features[f]]
        
        # Run clustering
        self.affective_clusterer.load_data()
        self.affective_clusterer.preprocess_data(exclude_features=['FinalGrade', 'Gender'])
        self.affective_clusterer.perform_kprototypes_clustering(n_clusters=n_clusters, n_init=5)
        profile_df = self.affective_clusterer.analyze_clusters()
        
        self.affective_labels = self.affective_clusterer.cluster_labels
        self.df['Affective_Cluster'] = self.affective_labels
        
        # Visualize engagement clusters
        self._visualize_cluster_view(
            'Affective',
            self.affective_clusterer.df_features[self.affective_clusterer.continuous_features].values,
            self.affective_labels,
            n_clusters
        )
        
        # Interpret affective levels
        affective_interpretations = self.interpret_affective(profile_df)
        
        return self.affective_labels, affective_interpretations

    def interpret_learning_styles(self, profile_df):
        """Interpret learning style clusters based on feature means."""
        interpretations = {}
        
        for _, row in profile_df.iterrows():
            cluster_id = row['Cluster']
            
            # Extract key metrics
            study_hours = float(row.get('StudyHours_mean', 0))
            online_courses = float(row.get('OnlineCourses_mean', 0))
            discussions_pct = float(row.get('Discussions_pct', '0').rstrip('%'))
            resources_pct = float(row.get('Resources_pct', '0').rstrip('%'))
            edutech_pct = float(row.get('EduTech_pct', '0').rstrip('%'))
            
            # Classify learning style
            if online_courses > 10 and edutech_pct > 70:
                style = "Digital Self-Learner"
                description = "Prefers online courses and technology-enhanced learning"
            elif discussions_pct > 60 and resources_pct > 60:
                style = "Collaborative Learner"
                description = "Actively participates in discussions and uses multiple resources"
            elif study_hours > 22:
                style = "Deep Independent Learner"
                description = "High study intensity, self-directed learning"
            else:
                style = "Traditional Learner"
                description = "Follows conventional learning approaches"
            
            interpretations[cluster_id] = {
                'Style': style,
                'Description': description,
                'Size': row['Size'],
                'Percentage': row['Percentage']
            }
        
        return interpretations
    
    def interpret_performance(self, profile_df):
        """Interpret performance clusters."""
        interpretations = {}
        
        for _, row in profile_df.iterrows():
            cluster_id = row['Cluster']
            
            exam_score = float(row.get('ExamScore_mean', 0))
            assignment = float(row.get('AssignmentCompletion_mean', 0))
            
            if exam_score >= 80 and assignment >= 80:
                level = "High Performer"
                description = "Consistently strong academic performance"
            elif exam_score >= 60 and assignment >= 60:
                level = "Moderate Performer"
                description = "Average academic performance with room for growth"
            else:
                level = "Struggling Performer"
                description = "Needs academic support and intervention"
            
            interpretations[cluster_id] = {
                'Level': level,
                'Description': description,
                'Avg_ExamScore': f"{exam_score:.1f}",
                'Avg_Assignment': f"{assignment:.1f}",
                'Size': row['Size'],
                'Percentage': row['Percentage']
            }
        
        return interpretations
    
    def interpret_affective(self, profile_df):
        """Interpret affective clusters."""
        interpretations = {}
        
        for _, row in profile_df.iterrows():
            cluster_id = row['Cluster']
            
            attendance = float(row.get('Attendance_mean', 0))
            motivation = int(row.get('Motivation_mode', 0))
            
            if attendance >= 85 and motivation >= 1:
                level = "Highly Engaged"
                description = "Strong attendance and motivation"
            elif attendance >= 70:
                level = "Moderately Engaged"
                description = "Decent affective with some variability"
            else:
                level = "Disengaged"
                description = "Low attendance and affective, needs intervention"
            
            interpretations[cluster_id] = {
                'Level': level,
                'Description': description,
                'Avg_Attendance': f"{attendance:.1f}%",
                'Size': row['Size'],
                'Percentage': row['Percentage']
            }
        
        return interpretations
    
    def create_composite_profiles(self, style_interp, perf_interp, eng_interp):
        """
        Create composite student profiles by combining all three dimensions.
        """
        print("\n" + "="*70)
        print("CREATING COMPOSITE STUDENT PROFILES")
        print("="*70)
        
        # Create composite labels
        self.df['Composite_Profile'] = (
            self.df['LearningStyle_Cluster'].astype(str) + '_' +
            self.df['Performance_Cluster'].astype(str) + '_' +
            self.df['Affective_Cluster'].astype(str)
        )
        
        # Analyze cross-tabulation
        print("\nCross-Tabulation: Learning Style vs Performance")
        style_perf = pd.crosstab(
            self.df['LearningStyle_Cluster'],
            self.df['Performance_Cluster'],
            normalize='index'
        ) * 100
        print(style_perf.round(1))
        
        print("\nCross-Tabulation: Learning Style vs Affective")
        style_eng = pd.crosstab(
            self.df['LearningStyle_Cluster'],
            self.df['Affective_Cluster'],
            normalize='index'
        ) * 100
        print(style_eng.round(1))
        
        print("\nCross-Tabulation: Performance vs Affective")
        perf_eng = pd.crosstab(
            self.df['Performance_Cluster'],
            self.df['Affective_Cluster'],
            normalize='index'
        ) * 100
        print(perf_eng.round(1))
        
        # Generate recommendations for each composite profile
        recommendations = self._generate_composite_recommendations(
            style_interp, perf_interp, eng_interp
        )
        
        return recommendations
    
    def _generate_composite_recommendations(self, style_interp, perf_interp, eng_interp):
        """Generate personalized recommendations based on composite profiles."""
        recommendations = {}
        
        for _, row in self.df.iterrows():
            style_id = row['LearningStyle_Cluster']
            perf_id = row['Performance_Cluster']
            eng_id = row['Affective_Cluster']
            profile = row['Composite_Profile']
            
            if profile not in recommendations:
                style = style_interp[style_id]['Style']
                performance = perf_interp[perf_id]['Level']
                affective = eng_interp[eng_id]['Level']
                
                # Generate tailored recommendations
                recs = []
                
                # Style-based recommendations
                if 'Digital' in style:
                    recs.append("Provide advanced online resources and MOOCs")
                elif 'Collaborative' in style:
                    recs.append("Facilitate group projects and peer learning")
                elif 'Deep Independent' in style:
                    recs.append("Offer challenging independent research projects")
                else:
                    recs.append("Maintain structured classroom instruction")
                
                # Performance-based recommendations
                if 'High' in performance:
                    recs.append("Offer advanced placement and enrichment opportunities")
                elif 'Struggling' in performance:
                    recs.append("Provide one-on-one tutoring and remedial support")
                else:
                    recs.append("Set incremental improvement goals")
                
                # Affective-based recommendations
                if 'Disengaged' in affective:
                    recs.append("Implement early intervention and motivation strategies")
                elif 'Moderately' in affective:
                    recs.append("Increase interactive and engaging content")
                else:
                    recs.append("Leverage as peer mentor and class leader")
                
                recommendations[profile] = {
                    'LearningStyle': style,
                    'Performance': performance,
                    'Affective': affective,
                    'Count': 1,
                    'Recommendations': recs
                }
            else:
                recommendations[profile]['Count'] += 1
        
        return recommendations
    
    def _visualize_cluster_view(self, view_name, data, labels, n_clusters):
        """
        Create scatter plot and silhouette visualization for a single view.
        
        Parameters:
        -----------
        view_name : str
            Name of the view (e.g., 'LearningStyle', 'Performance', 'Affective')
        data : ndarray
            Feature data for this view
        labels : ndarray
            Cluster labels
        n_clusters : int
            Number of clusters
        """
        from sklearn.metrics import silhouette_samples, silhouette_score
        from sklearn.decomposition import PCA
        
        print(f"\n  Creating visualizations for {view_name} view...")
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(data, labels)
        sample_silhouette_values = silhouette_samples(data, labels)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ===== Subplot 1: Silhouette plot =====
        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        
        y_lower = 10
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            # Aggregate silhouette scores for samples in cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.7,
                label=f'Cluster {i}'
            )
            
            # Label the silhouette plots with cluster numbers
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_xlabel('Silhouette Coefficient', fontsize=12)
        ax1.set_ylabel('Cluster', fontsize=12)
        ax1.set_title(f'{view_name} Silhouette Plot\nAvg Score: {silhouette_avg:.3f}', fontsize=14)
        ax1.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2, label='Average')
        ax1.set_yticks([])
        ax1.legend(loc='lower right')
        
        # ===== Subplot 2: Scatter plot (PCA if >2 dimensions) =====
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            variance_explained = pca.explained_variance_ratio_
            xlabel = f'PC1 ({variance_explained[0]:.1%} var)'
            ylabel = f'PC2 ({variance_explained[1]:.1%} var)'
            title = f'{view_name} Clusters (PCA)'
        elif data.shape[1] == 2:
            data_2d = data
            xlabel = 'Feature 1'
            ylabel = 'Feature 2'
            title = f'{view_name} Clusters'
        else:
            # 1D data - plot against index
            data_2d = np.column_stack([np.arange(len(data)), data])
            xlabel = 'Sample Index'
            ylabel = 'Feature Value'
            title = f'{view_name} Clusters'
        
        # Scatter plot
        for i in range(n_clusters):
            cluster_data = data_2d[labels == i]
            ax2.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                c=[colors[i]],
                label=f'Cluster {i}',
                alpha=0.6,
                edgecolors='k',
                linewidth=0.5,
                s=50
            )
        
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel(ylabel, fontsize=12)
        ax2.set_title(title, fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{view_name.lower()}_scatter_silhouette.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: visualizations/{view_name.lower()}_scatter_silhouette.png")
        plt.close()
    
    def visualize_multiview_results(self):
        """Create visualizations for multi-view clustering results."""
        print("\n" + "="*70)
        print("CREATING MULTI-VIEW VISUALIZATIONS")
        print("="*70)
        
        # 1. Distribution of each view
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        self.df['LearningStyle_Cluster'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0], color='skyblue'
        )
        axes[0].set_title('Learning Style Distribution', fontsize=14)
        axes[0].set_xlabel('Cluster', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        
        self.df['Performance_Cluster'].value_counts().sort_index().plot(
            kind='bar', ax=axes[1], color='lightcoral'
        )
        axes[1].set_title('Performance Distribution', fontsize=14)
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        
        self.df['Affective_Cluster'].value_counts().sort_index().plot(
            kind='bar', ax=axes[2], color='lightgreen'
        )
        axes[2].set_title('Affective Distribution', fontsize=14)
        axes[2].set_xlabel('Cluster', fontsize=12)
        axes[2].set_ylabel('Count', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('visualizations/multiview_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/multiview_distributions.png")
        
        # 2. Heatmap of cross-tabulations
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Style vs Performance
        style_perf = pd.crosstab(
            self.df['LearningStyle_Cluster'],
            self.df['Performance_Cluster']
        )
        sns.heatmap(style_perf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Learning Style vs Performance', fontsize=14)
        axes[0].set_xlabel('Performance Cluster', fontsize=12)
        axes[0].set_ylabel('Learning Style Cluster', fontsize=12)
        
        # Style vs Affective
        style_eng = pd.crosstab(
            self.df['LearningStyle_Cluster'],
            self.df['Affective_Cluster']
        )
        sns.heatmap(style_eng, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Learning Style vs Affective', fontsize=14)
        axes[1].set_xlabel('Affective Cluster', fontsize=12)
        axes[1].set_ylabel('Learning Style Cluster', fontsize=12)
        
        # Performance vs Affective
        perf_eng = pd.crosstab(
            self.df['Performance_Cluster'],
            self.df['Affective_Cluster']
        )
        sns.heatmap(perf_eng, annot=True, fmt='d', cmap='Oranges', ax=axes[2])
        axes[2].set_title('Performance vs Affective', fontsize=14)
        axes[2].set_xlabel('Affective Cluster', fontsize=12)
        axes[2].set_ylabel('Performance Cluster', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('visualizations/multiview_crosstabs.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/multiview_crosstabs.png")
    
    def save_results(self, recommendations):
        """Save multi-view clustering results."""
        # Save clustered data
        output_cols = ['LearningStyle_Cluster', 'Performance_Cluster', 
                      'Affective_Cluster', 'Composite_Profile']
        output_df = self.df[output_cols + list(self.df.columns.difference(output_cols))]
        output_df.to_csv('visualizations/multiview_student_profiles.csv', index=False)
        print("✓ Saved: visualizations/multiview_student_profiles.csv")
        
        # Save recommendations
        with open('visualizations/multiview_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        print("✓ Saved: visualizations/multiview_recommendations.json")
    
    def run_full_multiview_analysis(self):
        """Run complete multi-view clustering analysis."""
        print("\n" + "="*70)
        print("MULTI-VIEW LEARNING PATH CLUSTERING ANALYSIS")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Run three separate clusterings
        perf_labels, perf_interp = self.cluster_performance(n_clusters=3)
        style_labels, style_interp = self.cluster_learning_styles(n_clusters=4)
        eng_labels, eng_interp = self.cluster_affective(n_clusters=3)
        
        # Create composite profiles
        recommendations = self.create_composite_profiles(
            style_interp, perf_interp, eng_interp
        )
        
        # Visualize results
        self.visualize_multiview_results()
        
        # Save everything
        self.save_results(recommendations)
        
        # Print summary
        print("\n" + "="*70)
        print("MULTI-VIEW ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nLearning Style Clusters: {len(style_interp)}")
        for cid, info in style_interp.items():
            print(f"  Cluster {cid}: {info['Style']} ({info['Percentage']})")
        
        print(f"\nPerformance Clusters: {len(perf_interp)}")
        for cid, info in perf_interp.items():
            print(f"  Cluster {cid}: {info['Level']} ({info['Percentage']})")
        
        print(f"\nAffective Clusters: {len(eng_interp)}")
        for cid, info in eng_interp.items():
            print(f"  Cluster {cid}: {info['Level']} ({info['Percentage']})")
        
        print(f"\nComposite Profiles: {len(recommendations)}")
        print(f"Total Students: {len(self.df)}")
        
        return recommendations




    def predict_student_profile(self, student_features):
        """Predict the combined learning cluster for a new student."""
        # This function would take a new student's data, preprocess it according to each view's requirements,
        # and then use the trained clusterers to predict their cluster labels for each view.
        # Finally, it would combine those labels into a composite profile and return recommendations.

        style_data = {f: student_features[f] for f in self.learning_style_features if f in student_features}
        afffective_data = {f: student_features[f] for f in self.affective_features if f in student_features}
        performance_data = {f: student_features[f] for f in self.learning_performance_features if f in student_features}
        
        style_cluster = self.style_clusterer.predict_cluster(style_data)
        engage_cluster = self.affective_clusterer.predict_cluster(afffective_data)
        
        perf_cluster = self.performance_clusterer.predict_cluster(performance_data)

        composite_profile = [style_cluster, perf_cluster, engage_cluster]

        return composite_profile

    def save_model(self):
        """Save the trained clusterers for future use."""
        self.style_clusterer.save_model('models/learning_style_clusterer.pkl')
        self.performance_clusterer.save_model('models/performance_clusterer.pkl')
        self.affective_clusterer.save_model('models/affective_clusterer.pkl')
        print("✓ Saved all clusterers to 'models/' directory")

    def load_model(self):
        """Load the trained clusterers."""
        self.style_clusterer.load_model('models/learning_style_clusterer.pkl')
        self.performance_clusterer.load_model('models/performance_clusterer.pkl')
        self.affective_clusterer.load_model('models/affective_clusterer.pkl')
        print("✓ Loaded all clusterers from 'models/' directory")





def main():
    """Main execution function."""
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run multi-view clustering
    multiview = MultiViewLearningPathClusterer(
        data_path='data/training_files/student_performance.csv',
        use_minmax_scaler=True
    )
    
    recommendations = multiview.run_full_multiview_analysis()
    
    # Print sample recommendations
    print("\n" + "="*70)
    print("SAMPLE COMPOSITE RECOMMENDATIONS")
    print("="*70)
    
    for i, (profile, rec) in enumerate(list(recommendations.items())[:5]):
        print(f"\nProfile {i+1}: {profile}")
        print(f"  Learning Style: {rec['LearningStyle']}")
        print(f"  Performance: {rec['Performance']}")
        print(f"  Affective: {rec['Affective']}")
        print(f"  Students: {rec['Count']}")
        print(f"  Recommendations:")
        for r in rec['Recommendations']:

            print(f"    • {r}")

if __name__ == "__main__":
    main()


