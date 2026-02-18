"""
Example Usage: User Interaction Prediction Model
=================================================
This script demonstrates how to use the trained model for predictions.
"""

from user_interaction_predictor import UserInteractionPredictor
import json


def example_1_train_and_predict():
    """Example 1: Train model and make predictions for all users."""
    print("=" * 80)
    print("Example 1: Training Model and Making Predictions")
    print("=" * 80)
    
    # Initialize and train
    predictor = UserInteractionPredictor()
    predictor.train_model()
    
    # Make predictions for all users
    print("\n📊 User Engagement Predictions:")
    print("-" * 80)
    
    for user_id in predictor.analyzer.user_data.keys():
        engagement, proba, features = predictor.predict_user_engagement(user_id)
        
        # Format output with emoji indicators
        emoji = "🔥" if engagement == "High" else "📈" if engagement == "Medium" else "📉"
        
        print(f"\n{emoji} User {user_id}:")
        print(f"   Engagement: {engagement} (confidence: {proba.max():.1%})")
        print(f"   Interactions: {features['total_interactions']}")
        print(f"   Completion: {features['completion_rate']:.1%}")


def example_2_personalized_recommendations():
    """Example 2: Get personalized recommendations for specific users."""
    print("\n" + "=" * 80)
    print("Example 2: Personalized Content Recommendations")
    print("=" * 80)
    
    predictor = UserInteractionPredictor()
    predictor.train_model()
    
    # Select a few sample users
    sample_users = list(predictor.analyzer.user_data.keys())[:3]
    
    for user_id in sample_users:
        print(f"\n🎯 Recommendations for User {user_id}:")
        print("-" * 80)
        
        # Get engagement and recommendations
        engagement, _, features = predictor.predict_user_engagement(user_id)
        recommendations = predictor.get_recommendations(user_id, n_recommendations=5)
        
        print(f"Engagement Level: {engagement}")
        print(f"Learning Profile:")
        print(f"  - Total Activity: {features['total_interactions']} interactions")
        print(f"  - Completion Rate: {features['completion_rate']:.1%}")
        print(f"  - Primary Platform: {'Mobile' if features['mobile_usage_ratio'] > 0.5 else 'Web'}")
        
        print(f"\nTop 5 Recommended Items:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['item_id']}")
            print(f"     - Recommendation Score: {rec['score']:.2f}")
            print(f"     - Content Effectiveness: {rec['effectiveness']:.2f}")
            print(f"     - Popularity Rank: {rec['popularity']} users")


def example_3_engagement_insights():
    """Example 3: Analyze engagement patterns and feature importance."""
    print("\n" + "=" * 80)
    print("Example 3: Engagement Insights and Feature Analysis")
    print("=" * 80)
    
    predictor = UserInteractionPredictor()
    predictor.train_model()
    
    # Feature importance
    print("\n📈 Top 10 Factors Influencing Engagement:")
    print("-" * 80)
    importance_df = predictor.classifier.get_feature_importance()
    
    for idx, row in importance_df.head(10).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = "█" * bar_length
        print(f"{row['feature']:30s} {bar} {row['importance']:.3f}")
    
    # Engagement statistics
    print("\n📊 Engagement Distribution:")
    print("-" * 80)
    engagement_dist = predictor.engagement_labels.value_counts()
    total = len(predictor.engagement_labels)
    
    for level, count in engagement_dist.items():
        percentage = (count / total) * 100
        emoji = "🔥" if level == "High" else "📈" if level == "Medium" else "📉"
        print(f"{emoji} {level:10s}: {count:2d} users ({percentage:5.1f}%)")


def example_4_content_analysis():
    """Example 4: Analyze content effectiveness and popularity."""
    print("\n" + "=" * 80)
    print("Example 4: Content Performance Analysis")
    print("=" * 80)
    
    predictor = UserInteractionPredictor()
    predictor.train_model()
    
    # Most popular content
    print("\n🏆 Top 10 Most Popular Content Items:")
    print("-" * 80)
    content_popularity = sorted(
        predictor.recommender.content_popularity.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for i, (item_id, popularity) in enumerate(content_popularity, 1):
        effectiveness = predictor.recommender.content_effectiveness.get(item_id, 0)
        print(f"{i:2d}. {item_id:10s} - {popularity:3d} interactions "
              f"(effectiveness: {effectiveness:.2f})")
    
    # Most effective content
    print("\n⭐ Top 10 Most Effective Content Items:")
    print("-" * 80)
    content_effectiveness = sorted(
        predictor.recommender.content_effectiveness.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for i, (item_id, effectiveness) in enumerate(content_effectiveness, 1):
        popularity = predictor.recommender.content_popularity.get(item_id, 0)
        print(f"{i:2d}. {item_id:10s} - effectiveness: {effectiveness:.2f} "
              f"({popularity} users)")


def example_5_compare_users():
    """Example 5: Compare different user profiles side by side."""
    print("\n" + "=" * 80)
    print("Example 5: User Profile Comparison")
    print("=" * 80)
    
    predictor = UserInteractionPredictor()
    predictor.train_model()
    
    # Get users from different engagement levels
    high_users = [u for u, l in zip(predictor.analyzer.user_data.keys(), 
                                     predictor.engagement_labels) if l == 'High']
    medium_users = [u for u, l in zip(predictor.analyzer.user_data.keys(), 
                                       predictor.engagement_labels) if l == 'Medium']
    low_users = [u for u, l in zip(predictor.analyzer.user_data.keys(), 
                                    predictor.engagement_labels) if l == 'Low']
    
    comparison_users = [
        high_users[0] if high_users else None,
        medium_users[0] if medium_users else None,
        low_users[0] if low_users else None
    ]
    
    print("\n📊 Side-by-Side Comparison:")
    print("-" * 80)
    print(f"{'Metric':<30} {'High':>15} {'Medium':>15} {'Low':>15}")
    print("-" * 80)
    
    metrics = [
        ('Engagement Level', lambda f: 'High/Medium/Low'),
        ('Total Interactions', lambda f: f['total_interactions']),
        ('Duration (hours)', lambda f: f'{f["duration_hours"]:.1f}'),
        ('Sessions', lambda f: f['num_sessions']),
        ('Completion Rate', lambda f: f'{f["completion_rate"]:.1%}'),
        ('Questions Attempted', lambda f: f['num_questions_attempted']),
        ('Video Engagement', lambda f: f['video_engagement']),
        ('Audio Engagement', lambda f: f['audio_engagement']),
        ('Mobile Usage', lambda f: f'{f["mobile_usage_ratio"]:.1%}'),
    ]
    
    user_features = []
    for user_id in comparison_users:
        if user_id:
            _, _, features = predictor.predict_user_engagement(user_id)
            user_features.append(features)
        else:
            user_features.append(None)
    
    print(f"{'User ID':<30} {comparison_users[0] if comparison_users[0] else 'N/A':>15} "
          f"{comparison_users[1] if comparison_users[1] else 'N/A':>15} "
          f"{comparison_users[2] if comparison_users[2] else 'N/A':>15}")
    print("-" * 80)
    
    for metric_name, metric_func in metrics:
        if metric_name == 'Engagement Level':
            print(f"{metric_name:<30} {'High':>15} {'Medium':>15} {'Low':>15}")
        else:
            values = []
            for features in user_features:
                if features:
                    try:
                        value = metric_func(features)
                        values.append(f"{value:>15}")
                    except:
                        values.append(f"{'N/A':>15}")
                else:
                    values.append(f"{'N/A':>15}")
            print(f"{metric_name:<30} {values[0]} {values[1]} {values[2]}")


def example_6_save_and_load():
    """Example 6: Save and load the trained model."""
    print("\n" + "=" * 80)
    print("Example 6: Model Persistence")
    print("=" * 80)
    
    # Train and save
    print("\n💾 Training and saving model...")
    predictor = UserInteractionPredictor()
    predictor.train_model()
    predictor.save_model("models/user_interaction_model.pkl")
    print("✓ Model saved successfully!")
    
    # Note: Loading would require the model to be properly saved first
    print("\n📂 To load a saved model in the future:")
    print("   predictor = UserInteractionPredictor()")
    print("   predictor.load_model('models/user_interaction_model.pkl')")
    
    # Generate reports
    print("\n📄 Generating comprehensive reports...")
    predictor.generate_report("user_interaction_report.txt")
    print("✓ Report generated successfully!")


def main():
    """Run all examples."""
    print("\n" + "🚀" * 40)
    print("User Interaction Prediction Model - Example Usage")
    print("🚀" * 40)
    
    # Run examples
    example_1_train_and_predict()
    example_2_personalized_recommendations()
    example_3_engagement_insights()
    example_4_content_analysis()
    example_5_compare_users()
    example_6_save_and_load()
    
    print("\n" + "✅" * 40)
    print("All Examples Completed Successfully!")
    print("✅" * 40)
    
    print("\n📚 Next Steps:")
    print("   1. Review the generated visualizations in visualizations/")
    print("   2. Check user_profiles.json for detailed user data")
    print("   3. Read user_interaction_report.txt for comprehensive analysis")
    print("   4. Customize the model for your specific use case")
    print("   5. Integrate predictions into your application")


if __name__ == "__main__":
    main()
