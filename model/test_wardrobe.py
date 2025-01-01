import json
from pathlib import Path

# Test wardrobe data with realistic items
test_wardrobe = {
    "tshirt_white_1": {
        "category": "T-shirt",
        "color_properties": {
            "primary": {
                "rgb": [255, 255, 255],
                "name": "white",
                "temperature": "neutral",
                "is_light": True
            }
        },
        "attributes": ["casual", "basic", "lightweight", "cotton"],
        "pattern_info": {
            "primary_pattern": "solid",
            "has_print": False
        },
        "image_path": "test_images/tshirt_white_1.jpg",
        "seasonal_score": {
            "summer": 0.9,
            "spring": 0.8,
            "fall": 0.5,
            "winter": 0.3
        }
    },
    
    "jeans_blue_1": {
        "category": "Jeans",
        "color_properties": {
            "primary": {
                "rgb": [0, 0, 139],
                "name": "navy",
                "temperature": "cool",
                "is_light": False
            }
        },
        "attributes": ["casual", "denim", "versatile"],
        "pattern_info": {
            "primary_pattern": "solid",
            "has_print": False
        },
        "image_path": "test_images/jeans_blue_1.jpg",
        "seasonal_score": {
            "summer": 0.6,
            "spring": 0.8,
            "fall": 0.9,
            "winter": 0.7
        }
    },
    
    "sweater_black_1": {
        "category": "Sweater",
        "color_properties": {
            "primary": {
                "rgb": [0, 0, 0],
                "name": "black",
                "temperature": "neutral",
                "is_light": False
            }
        },
        "attributes": ["warm", "cozy", "knit", "heavyweight"],
        "pattern_info": {
            "primary_pattern": "knit",
            "has_print": False
        },
        "image_path": "test_images/sweater_black_1.jpg",
        "seasonal_score": {
            "summer": 0.1,
            "spring": 0.4,
            "fall": 0.9,
            "winter": 1.0
        }
    },
    
    "shorts_khaki_1": {
        "category": "Shorts",
        "color_properties": {
            "primary": {
                "rgb": [189, 183, 107],
                "name": "khaki",
                "temperature": "warm",
                "is_light": True
            }
        },
        "attributes": ["casual", "lightweight", "summer", "cotton"],
        "pattern_info": {
            "primary_pattern": "solid",
            "has_print": False
        },
        "image_path": "test_images/shorts_khaki_1.jpg",
        "seasonal_score": {
            "summer": 1.0,
            "spring": 0.8,
            "fall": 0.3,
            "winter": 0.0
        }
    },
    
    "dress_floral_1": {
        "category": "Dress",
        "color_properties": {
            "primary": {
                "rgb": [255, 182, 193],
                "name": "light_pink",
                "temperature": "warm",
                "is_light": True
            },
            "secondary": {
                "rgb": [144, 238, 144],
                "name": "light_green",
                "temperature": "cool",
                "is_light": True
            }
        },
        "attributes": ["feminine", "floral", "lightweight", "summer"],
        "pattern_info": {
            "primary_pattern": "floral",
            "has_print": True
        },
        "image_path": "test_images/dress_floral_1.jpg",
        "seasonal_score": {
            "summer": 0.9,
            "spring": 1.0,
            "fall": 0.4,
            "winter": 0.1
        }
    },
    
    "blazer_navy_1": {
        "category": "Blazer",
        "color_properties": {
            "primary": {
                "rgb": [0, 0, 128],
                "name": "navy",
                "temperature": "cool",
                "is_light": False
            }
        },
        "attributes": ["formal", "structured", "professional", "lined"],
        "pattern_info": {
            "primary_pattern": "solid",
            "has_print": False
        },
        "image_path": "test_images/blazer_navy_1.jpg",
        "seasonal_score": {
            "summer": 0.4,
            "spring": 0.7,
            "fall": 0.9,
            "winter": 0.8
        }
    }
}

def create_test_wardrobe():
    """Create a test wardrobe JSON file"""
    # Ensure the test_data directory exists
    Path("test_data").mkdir(exist_ok=True)
    
    # Save the test wardrobe
    with open("test_data/test_wardrobe.json", "w") as f:
        json.dump(test_wardrobe, f, indent=4)

def load_test_wardrobe():
    """Load the test wardrobe data"""
    if not Path("test_data/test_wardrobe.json").exists():
        create_test_wardrobe()
    
    with open("test_data/test_wardrobe.json", "r") as f:
        return json.load(f)

if __name__ == "__main__":
    create_test_wardrobe() 