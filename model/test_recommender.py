from outfit_recommender import OutfitRecommender, WardrobeItem
from test_wardrobe import load_test_wardrobe
import json
from datetime import datetime

def print_outfit(outfit):
    """Pretty print an outfit"""
    print("\nOutfit Details:")
    print("-" * 50)
    for item_type, item in outfit.items():
        print(f"\n{item_type.title()}:")
        print(f"  - Category: {item.category}")
        print(f"  - Color: {next(iter(item.color_properties.values()))['name']}")
        print(f"  - Attributes: {', '.join(item.attributes)}")
        if item.seasonal_score:
            best_season = max(item.seasonal_score.items(), key=lambda x: x[1])[0]
            print(f"  - Best Season: {best_season}")

def test_recommendations():
    """Test the outfit recommender with our test wardrobe"""
    # Load test data
    test_data = load_test_wardrobe()
    
    # Initialize recommender
    recommender = OutfitRecommender()
    
    # Add test items to wardrobe
    for item_id, item_data in test_data.items():
        recommender.wardrobe.items[item_id] = WardrobeItem(
            item_id=item_id,
            **item_data
        )
    
    # Get current season
    current_month = datetime.now().month
    season_map = {
        (12, 1, 2): 'winter',
        (3, 4, 5): 'spring',
        (6, 7, 8): 'summer',
        (9, 10, 11): 'fall'
    }
    current_season = next(season for months, season in season_map.items() if current_month in months)
    print(f"\nCurrent season: {current_season}")
    
    print("\nTesting Casual Summer Outfit (Starting with T-shirt):")
    outfits = recommender.get_outfit_recommendations("tshirt_white_1")
    print_outfit(outfits[0])
    
    print("\nTesting Casual Winter Outfit (Starting with Sweater):")
    outfits = recommender.get_outfit_recommendations("sweater_black_1")
    print_outfit(outfits[0])
    
    print("\nTesting Formal Outfit (Starting with Blazer):")
    outfits = recommender.get_outfit_recommendations("blazer_navy_1", style="formal")
    print_outfit(outfits[0])
    
    print("\nTesting Business Casual Outfit (Starting with Blazer):")
    outfits = recommender.get_outfit_recommendations("blazer_navy_1", style="business_casual")
    print_outfit(outfits[0])
    
    print("\nTesting Summer Casual with Shorts:")
    outfits = recommender.get_outfit_recommendations("shorts_khaki_1")
    print_outfit(outfits[0])
    
    print("\nTesting Dress-based Outfit:")
    outfits = recommender.get_outfit_recommendations("dress_floral_1")
    print_outfit(outfits[0])

if __name__ == "__main__":
    test_recommendations() 