from typing import Dict, List, Optional, TypedDict, Union
import colorsys
import logging
import json

class FrontendColor(TypedDict):
    name: str
    hex: str

class PredictionResponse(TypedDict):
    category: str
    colors: List[FrontendColor]
    materials: List[str]
    occasions: List[str]
    seasons: List[str]
    confidence: float
    pattern: str

# Category mapping from DeepFashion to frontend categories
CATEGORY_MAPPING = {
    # Tops
    'Tee': 'Tops',
    'Tank': 'Tops',
    'Blouse': 'Tops',
    'Shirt': 'Tops',
    'Top': 'Tops',
    'Sweater': 'Tops',
    
    # Robes
    'Dress': 'Robes',
    'Gown': 'Robes',
    
    # Pantalons
    'Jeans': 'Pantalons',
    'Pants': 'Pantalons',
    'Shorts': 'Pantalons',
    'Leggings': 'Pantalons',
    
    # Jupes
    'Skirt': 'Jupes',
    
    # Vêtements d'extérieur
    'Jacket': 'Vêtements d\'extérieur',
    'Coat': 'Vêtements d\'extérieur',
    'Cardigan': 'Vêtements d\'extérieur',
    'Blazer': 'Vêtements d\'extérieur',
    
    # Chaussures
    'Shoes': 'Chaussures',
    'Boots': 'Chaussures',
    'Sandals': 'Chaussures',
    'Sneakers': 'Chaussures',
    
    # Sacs
    'Bag': 'Sacs',
    'Backpack': 'Sacs',
    'Purse': 'Sacs',
    
    # Chapeaux
    'Hat': 'Chapeaux',
    'Cap': 'Chapeaux',
    
    # Bijoux
    'Jewelry': 'Bijoux',
    'Necklace': 'Bijoux',
    'Bracelet': 'Bijoux',
    'Ring': 'Bijoux',
    
    # Default
    'Other': 'Autres Articles'
}

# Color mapping from model colors to frontend colors
COLOR_MAPPING = {
    # Basic Colors (matching frontend exactly)
    'white': {'name': 'Blanc', 'hex': '#FFFFFF'},
    'ivory': {'name': 'Ivoire', 'hex': '#FFFFF0'},
    'beige': {'name': 'Beige', 'hex': '#F5F5DC'},
    'light_gray': {'name': 'Gris Clair', 'hex': '#D3D3D3'},
    'dark_gray': {'name': 'Gris Foncé', 'hex': '#696969'},
    'black': {'name': 'Noir', 'hex': '#000000'},
    'light_yellow': {'name': 'Jaune Clair', 'hex': '#FFFFE0'},
    'yellow': {'name': 'Jaune', 'hex': '#FFFF00'},
    'turmeric': {'name': 'Couleur Curcuma', 'hex': '#FFA500'},
    'orange': {'name': 'Orange', 'hex': '#FFA500'},
    'coral': {'name': 'Corail', 'hex': '#FF7F50'},
    'red': {'name': 'Rouge', 'hex': '#FF0000'},
    'pink': {'name': 'Rose', 'hex': '#FFC0CB'},
    'hot_pink': {'name': 'Rose Vif', 'hex': '#FF69B4'},
    'light_green': {'name': 'Vert Clair', 'hex': '#90EE90'},
    'green': {'name': 'Vert', 'hex': '#008000'},
    'olive': {'name': 'Olive', 'hex': '#808000'},
    'dark_olive': {'name': 'Olive Foncé', 'hex': '#556B2F'},
    'teal': {'name': 'Sarcelle', 'hex': '#008080'},
    'khaki': {'name': 'Kaki', 'hex': '#F0E68C'},
    'cyan': {'name': 'Cyan', 'hex': '#00FFFF'},
    'sky_blue': {'name': 'Bleu Ciel', 'hex': '#87CEEB'},
    'blue': {'name': 'Bleu', 'hex': '#0000FF'},
    'navy': {'name': 'Marine', 'hex': '#000080'},
    'lavender': {'name': 'Lavande', 'hex': '#E6E6FA'},
    'purple': {'name': 'Violet', 'hex': '#800080'},
    'burgundy': {'name': 'Bourgogne', 'hex': '#800020'},
    'camel': {'name': 'Camel', 'hex': '#C19A6B'},
    'brown': {'name': 'Brun', 'hex': '#964B00'},
    'dark_brown': {'name': 'Marron Foncé', 'hex': '#654321'},
    'magenta': {'name': 'Magenta', 'hex': '#FF00FF'},
    'gold': {'name': 'Or', 'hex': '#FFD700'},
    'silver': {'name': 'Argent', 'hex': '#C0C0C0'},
    
    # Additional mappings for color analyzer detection
    'charcoal': {'name': 'Gris Foncé', 'hex': '#696969'},  # Map to existing dark gray
    'light_blue': {'name': 'Bleu Ciel', 'hex': '#87CEEB'},  # Map to existing sky blue
    'taupe': {'name': 'Gris Foncé', 'hex': '#696969'},  # Map to existing dark gray
    'tan': {'name': 'Camel', 'hex': '#C19A6B'},  # Map to existing camel
    'turquoise': {'name': 'Cyan', 'hex': '#00FFFF'},  # Map to existing cyan
    'mint': {'name': 'Vert Clair', 'hex': '#90EE90'},  # Map to existing light green
    'gray': {'name': 'Gris Clair', 'hex': '#D3D3D3'},  # Map to existing light gray
}

# Attribute to material mapping
MATERIAL_MAPPING = {
    'cotton': 'Coton',
    'linen': 'Lin',
    'wool': 'Laine',
    'silk': 'Soie',
    'polyester': 'Polyester',
    'nylon': 'Nylon',
    'leather': 'Cuir',
    'suede': 'Daim',
    'denim': 'Denim',
    'velvet': 'Velours',
    'satin': 'Satin',
    'jersey': 'Jersey',
    'cashmere': 'Cachemire',
    'flannel': 'Flanelle',
    'knit': 'Maille',
    'fleece': 'Molleton',
    'tweed': 'Tweed',
    'neoprene': 'Néoprène',
    'tulle': 'Tulle',
    'lace': 'Dentelle',
    'chiffon': 'Mousseline',
    'viscose': 'Viscose',
    'spandex': 'Élasthanne'
}

def get_season_from_attributes(attributes: List[Dict[str, float]]) -> List[str]:
    """Determine seasons based on attributes."""
    seasons = []
    
    # Example logic - you should adjust based on your actual attributes
    for attr in attributes:
        if 'light' in attr['attribute'].lower() or 'thin' in attr['attribute'].lower():
            seasons.extend(['Printemps', 'Été'])
        if 'thick' in attr['attribute'].lower() or 'warm' in attr['attribute'].lower():
            seasons.extend(['Automne', 'Hiver'])
    
    # Remove duplicates
    return list(set(seasons))

def get_occasions_from_attributes(attributes: List[Dict[str, float]]) -> List[str]:
    """Determine occasions based on attributes."""
    occasions = []
    
    # Map attributes to occasions
    attribute_to_occasion = {
        'casual': 'Quotidien',
        'formal': 'Formel',
        'business': 'Travail',
        'party': 'Fête',
        'sports': 'Sport',
        'beach': 'Plage',
        'travel': 'Voyage',
        'home': 'Maison'
    }
    
    for attr in attributes:
        attr_name = attr['attribute'].lower()
        for key, occasion in attribute_to_occasion.items():
            if key in attr_name:
                occasions.append(occasion)
    
    # If no specific occasions found, default to 'Quotidien'
    if not occasions:
        occasions = ['Quotidien']
    
    return list(set(occasions))

def map_prediction_to_frontend(prediction_results: Dict) -> PredictionResponse:
    """
    Map the model's prediction results to the frontend's expected format.
    """
    # Log raw predictions
    logging.info("Raw prediction results:")
    logging.info(f"Categories: {json.dumps(prediction_results['top_categories'], indent=2)}")
    logging.info(f"Attributes: {json.dumps(prediction_results['attributes'], indent=2)}")
    logging.info(f"Colors: {json.dumps(prediction_results['dominant_colors'], indent=2)}")
    
    # Get the primary category
    category = prediction_results['top_categories'][0]['category']
    frontend_category = CATEGORY_MAPPING.get(category, 'Autres Articles')
    
    # Map colors
    frontend_colors = []
    for color in prediction_results['dominant_colors']:
        if color['percentage'] > 5:  # Only include colors with >5% presence
            mapped_color = COLOR_MAPPING.get(color['color_name'], None)
            if mapped_color:
                frontend_colors.append(mapped_color)
    
    # If no colors mapped, add a default
    if not frontend_colors:
        frontend_colors = [{'name': 'Noir', 'hex': '#000000'}]
    
    # Get materials from attributes
    materials = []
    for attr in prediction_results['attributes']:
        attr_name = attr['attribute'].lower()
        for material_key, material_name in MATERIAL_MAPPING.items():
            if material_key in attr_name:
                materials.append(material_name)
    
    # If no materials detected, add default
    if not materials:
        materials = ['Autres Matériaux']
    
    # Get seasons and occasions
    seasons = get_season_from_attributes(prediction_results['attributes'])
    occasions = get_occasions_from_attributes(prediction_results['attributes'])
    
    # Get pattern information
    pattern = prediction_results.get('pattern_analysis', {}).get('primary_pattern', 'solid')
    
    # Prepare frontend response
    frontend_response = {
        'category': frontend_category,
        'colors': frontend_colors,
        'materials': materials,
        'occasions': occasions,
        'seasons': seasons,
        'confidence': prediction_results['top_categories'][0]['confidence'],
        'pattern': pattern
    }
    
    # Log frontend response
    logging.info("Frontend response:")
    logging.info(json.dumps(frontend_response, indent=2))
    
    return frontend_response 