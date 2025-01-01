import numpy as np
from typing import List, Dict, Tuple
import colorsys
import json
import os
from datetime import datetime

class WardrobeItem:
    def __init__(self, 
                 item_id: str,
                 category: str,
                 color_properties: Dict,
                 attributes: List[str],
                 pattern_info: Dict,
                 image_path: str,
                 seasonal_score: Dict = None):
        self.item_id = item_id
        self.category = category
        self.color_properties = color_properties
        self.attributes = attributes
        self.pattern_info = pattern_info
        self.image_path = image_path
        self.seasonal_score = seasonal_score or {}
        
        # Classify the item type
        self.item_type = self._classify_item_type(category)
    
    def _classify_item_type(self, category: str) -> str:
        """Classify the item into basic types (top, bottom, dress, etc.)"""
        category_lower = category.lower()
        if any(word in category_lower for word in ['shirt', 'blouse', 'top', 'sweater', 't-shirt']):
            return 'top'
        elif any(word in category_lower for word in ['pants', 'jeans', 'skirt', 'shorts']):
            return 'bottom'
        elif any(word in category_lower for word in ['dress', 'gown']):
            return 'dress'
        elif any(word in category_lower for word in ['jacket', 'coat', 'blazer']):
            return 'outerwear'
        elif any(word in category_lower for word in ['shoes', 'boots', 'sneakers']):
            return 'shoes'
        elif any(word in category_lower for word in ['bag', 'purse']):
            return 'bag'
        elif any(word in category_lower for word in ['necklace', 'earrings', 'bracelet']):
            return 'accessories'
        return 'other'
    
    def get_best_season(self) -> str:
        """Get the best season for this item based on its seasonal scores"""
        if not self.seasonal_score:
            # Determine season based on attributes and properties
            scores = {
                'summer': 0,
                'spring': 0,
                'fall': 0,
                'winter': 0
            }
            
            # Check attributes
            summer_keywords = ['lightweight', 'breathable', 'summer', 'thin']
            winter_keywords = ['warm', 'cozy', 'heavyweight', 'winter', 'thick']
            spring_keywords = ['light', 'spring', 'moderate']
            fall_keywords = ['fall', 'autumn', 'moderate']
            
            for attr in self.attributes:
                attr = attr.lower()
                if attr in summer_keywords:
                    scores['summer'] += 1
                if attr in winter_keywords:
                    scores['winter'] += 1
                if attr in spring_keywords:
                    scores['spring'] += 1
                if attr in fall_keywords:
                    scores['fall'] += 1
            
            # Check color properties
            primary_color = next(iter(self.color_properties.values()))
            if primary_color.get('is_light', False):
                scores['summer'] += 0.5
                scores['spring'] += 0.5
            else:
                scores['winter'] += 0.5
                scores['fall'] += 0.5
            
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return max(self.seasonal_score.items(), key=lambda x: x[1])[0]

class Wardrobe:
    def __init__(self, wardrobe_path: str = 'wardrobe.json'):
        self.wardrobe_path = wardrobe_path
        self.items: Dict[str, WardrobeItem] = {}
        self.load_wardrobe()
    
    def load_wardrobe(self):
        """Load wardrobe from JSON file"""
        if os.path.exists(self.wardrobe_path):
            with open(self.wardrobe_path, 'r') as f:
                wardrobe_data = json.load(f)
                for item_id, item_data in wardrobe_data.items():
                    self.items[item_id] = WardrobeItem(
                        item_id=item_id,
                        category=item_data['category'],
                        color_properties=item_data['color_properties'],
                        attributes=item_data['attributes'],
                        pattern_info=item_data['pattern_info'],
                        image_path=item_data['image_path']
                    )
    
    def save_wardrobe(self):
        """Save wardrobe to JSON file"""
        wardrobe_data = {
            item_id: {
                'category': item.category,
                'color_properties': item.color_properties,
                'attributes': item.attributes,
                'pattern_info': item.pattern_info,
                'image_path': item.image_path
            }
            for item_id, item in self.items.items()
        }
        with open(self.wardrobe_path, 'w') as f:
            json.dump(wardrobe_data, f, indent=4)
    
    def add_item(self, item: WardrobeItem):
        """Add a new item to the wardrobe"""
        self.items[item.item_id] = item
        self.save_wardrobe()
    
    def remove_item(self, item_id: str):
        """Remove an item from the wardrobe"""
        if item_id in self.items:
            del self.items[item_id]
            self.save_wardrobe()
    
    def get_items_by_type(self, item_type: str) -> List[WardrobeItem]:
        """Get all items of a specific type"""
        return [item for item in self.items.values() if item.item_type == item_type]

class OutfitRecommender:
    def __init__(self):
        # Initialize wardrobe
        self.wardrobe = Wardrobe()
        
        # Define style categories with their characteristics
        self.style_categories = {
            'casual': {
                'attributes': ['casual', 'comfortable', 'relaxed', 'basic'],
                'pieces': ['t-shirt', 'jeans', 'sneakers', 'sweater'],
                'occasions': ['weekend', 'daily', 'outdoor'],
                'formality_level': 1
            },
            'formal': {
                'attributes': ['formal', 'structured', 'professional', 'elegant'],
                'pieces': ['suit', 'dress', 'blazer', 'slacks'],
                'occasions': ['business', 'ceremony', 'evening'],
                'formality_level': 3
            },
            'business_casual': {
                'attributes': ['smart', 'professional', 'neat', 'polished'],
                'pieces': ['button-up', 'khakis', 'blouse', 'skirt'],
                'occasions': ['office', 'meeting', 'presentation'],
                'formality_level': 2
            }
        }
        
        # Define outfit templates with seasonal appropriateness
        self.outfit_templates = {
            'casual_summer': {
                'required': ['top', 'bottom', 'shoes'],
                'optional': ['accessories'],
                'preferred_attributes': ['lightweight', 'breathable', 'casual'],
                'avoided_attributes': ['warm', 'heavyweight', 'formal']
            },
            'casual_winter': {
                'required': ['top', 'bottom', 'shoes', 'outerwear'],
                'optional': ['accessories'],
                'preferred_attributes': ['warm', 'cozy', 'casual'],
                'avoided_attributes': ['lightweight', 'thin']
            },
            'formal': {
                'required': ['top', 'bottom', 'shoes'],
                'optional': ['outerwear', 'accessories', 'bag'],
                'alternatives': ['dress'],
                'preferred_attributes': ['formal', 'structured', 'professional'],
                'avoided_attributes': ['casual', 'sporty']
            },
            'business_casual': {
                'required': ['top', 'bottom', 'shoes'],
                'optional': ['outerwear', 'accessories'],
                'alternatives': ['dress'],
                'preferred_attributes': ['professional', 'smart', 'neat'],
                'avoided_attributes': ['sporty', 'beachwear']
            }
        }
    
    def get_style_recommendation(self, 
                               categories: List[str], 
                               attributes: List[str], 
                               color_properties: Dict,
                               pattern_info: Dict) -> Dict:
        """Generate style recommendations based on item analysis."""
        # Calculate style scores
        style_scores = {style: 0.0 for style in self.style_categories.keys()}
        
        # Score based on attributes
        for style, info in self.style_categories.items():
            # Match attributes
            attribute_matches = sum(1 for attr in attributes 
                                 if any(style_attr in attr.lower() 
                                      for style_attr in info['attributes']))
            style_scores[style] += attribute_matches * 2
            
            # Match categories/pieces
            category_matches = sum(1 for cat in categories 
                                if any(piece in cat.lower() 
                                     for piece in info['pieces']))
            style_scores[style] += category_matches * 1.5
        
        # Get primary and secondary styles
        sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
        primary_style = sorted_styles[0][0]
        secondary_style = sorted_styles[1][0] if sorted_styles[1][1] > 0 else None
        
        # Determine formality level
        formality_level = self.style_categories[primary_style]['formality_level']
        
        # Get suitable occasions based on primary and secondary styles
        occasions = set()
        occasions.update(self.style_categories[primary_style]['occasions'])
        if secondary_style:
            occasions.update(self.style_categories[secondary_style]['occasions'])
        
        # Consider pattern for occasion appropriateness
        pattern = pattern_info.get('primary_pattern', 'solid')
        if pattern == 'solid':
            occasions.update(['formal', 'business'])
        elif pattern in ['floral', 'print']:
            occasions.add('casual')
        
        # Get color temperature and brightness
        primary_color = next(iter(color_properties.values()))
        is_light = primary_color.get('is_light', False)
        temperature = primary_color.get('temperature', 'neutral')
        
        return {
            'primary_style': primary_style,
            'secondary_style': secondary_style,
            'formality_level': formality_level,
            'suitable_occasions': list(occasions),
            'color_characteristics': {
                'is_light': is_light,
                'temperature': temperature
            }
        }
    
    def _calculate_color_compatibility(self, item1: WardrobeItem, item2: WardrobeItem) -> float:
        """Calculate color compatibility score between two items"""
        # Get primary colors
        color1 = next(iter(item1.color_properties.values()))
        color2 = next(iter(item2.color_properties.values()))
        
        # Convert to HSV for better comparison
        rgb1 = [x/255.0 for x in color1.get('rgb', [0, 0, 0])]
        rgb2 = [x/255.0 for x in color2.get('rgb', [0, 0, 0])]
        
        h1, s1, v1 = colorsys.rgb_to_hsv(*rgb1)
        h2, s2, v2 = colorsys.rgb_to_hsv(*rgb2)
        
        # Calculate compatibility scores
        hue_diff = min(abs(h1 - h2), 1 - abs(h1 - h2))
        saturation_compatibility = 1 - abs(s1 - s2)
        value_compatibility = 1 - abs(v1 - v2)
        
        # Combine scores (weighted average)
        return (hue_diff * 0.5 + saturation_compatibility * 0.3 + value_compatibility * 0.2)
    
    def _get_matching_items(self, 
                          base_item: WardrobeItem, 
                          target_type: str,
                          season: str,
                          style: str,
                          min_compatibility: float = 0.6) -> List[WardrobeItem]:
        """Get matching items of a specific type that go well with the base item"""
        matching_items = []
        candidates = self.wardrobe.get_items_by_type(target_type)
        
        template = self.outfit_templates[style]
        preferred_attrs = template['preferred_attributes']
        avoided_attrs = template['avoided_attributes']
        
        for item in candidates:
            # Calculate base compatibility
            color_score = self._calculate_color_compatibility(base_item, item)
            
            # Calculate seasonal appropriateness
            seasonal_score = item.seasonal_score.get(season, 0.5)
            
            # Calculate style appropriateness
            style_score = 0.5
            item_attrs = [attr.lower() for attr in item.attributes]
            preferred_matches = sum(1 for attr in preferred_attrs if attr in item_attrs)
            avoided_matches = sum(1 for attr in avoided_attrs if attr in item_attrs)
            if preferred_matches > 0:
                style_score += 0.25
            if avoided_matches == 0:
                style_score += 0.25
            
            # Combined score
            total_score = (color_score * 0.4 + seasonal_score * 0.3 + style_score * 0.3)
            
            if total_score >= min_compatibility:
                matching_items.append((item, total_score))
        
        # Sort by total score
        matching_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in matching_items]
    
    def generate_outfit(self, base_item: WardrobeItem, style: str) -> Dict[str, WardrobeItem]:
        """Generate a complete outfit based on a base item and style"""
        # Determine season based on current date and base item
        current_month = datetime.now().month
        season_map = {
            (12, 1, 2): 'winter',
            (3, 4, 5): 'spring',
            (6, 7, 8): 'summer',
            (9, 10, 11): 'fall'
        }
        current_season = next(season for months, season in season_map.items() if current_month in months)
        
        # Map basic styles to seasonal styles
        if style == 'casual':
            if current_season in ['summer', 'spring']:
                style = 'casual_summer'
            else:
                style = 'casual_winter'
        
        outfit = {}
        template = self.outfit_templates[style]
        
        # Add base item to outfit
        outfit[base_item.item_type] = base_item
        
        # Fill required items
        for required_type in template['required']:
            if required_type not in outfit:
                matching_items = self._get_matching_items(
                    base_item, 
                    required_type,
                    current_season,
                    style
                )
                if matching_items:
                    outfit[required_type] = matching_items[0]
        
        # Add optional items if available
        for optional_type in template.get('optional', []):
            matching_items = self._get_matching_items(
                base_item,
                optional_type,
                current_season,
                style
            )
            if matching_items:
                outfit[optional_type] = matching_items[0]
        
        return outfit
    
    def get_outfit_recommendations(self, base_item_id: str, style: str = None) -> List[Dict[str, WardrobeItem]]:
        """Get multiple outfit recommendations based on a base item"""
        if base_item_id not in self.wardrobe.items:
            raise ValueError(f"Item {base_item_id} not found in wardrobe")
        
        base_item = self.wardrobe.items[base_item_id]
        
        # If style is not specified, determine based on item attributes
        if style is None:
            # Simple style detection based on attributes
            if any(attr in base_item.attributes for attr in ['formal', 'structured', 'professional']):
                style = 'formal'
            elif any(attr in base_item.attributes for attr in ['business', 'smart', 'professional']):
                style = 'business_casual'
            else:
                # Get current season for casual style
                current_month = datetime.now().month
                season_map = {
                    (12, 1, 2): 'winter',
                    (3, 4, 5): 'spring',
                    (6, 7, 8): 'summer',
                    (9, 10, 11): 'fall'
                }
                current_season = next(season for months, season in season_map.items() if current_month in months)
                style = f'casual_{current_season}' if current_season in ['summer', 'winter'] else 'casual_summer'
        
        # Generate primary outfit
        primary_outfit = self.generate_outfit(base_item, style)
        
        # Generate alternative outfits with different combinations
        alternative_outfits = []
        for item_type in primary_outfit:
            if item_type != base_item.item_type:
                matching_items = self._get_matching_items(
                    base_item,
                    item_type,
                    base_item.get_best_season(),
                    style
                )
                for alt_item in matching_items[1:3]:  # Get 2 alternatives
                    alt_outfit = primary_outfit.copy()
                    alt_outfit[item_type] = alt_item
                    alternative_outfits.append(alt_outfit)
        
        return [primary_outfit] + alternative_outfits 