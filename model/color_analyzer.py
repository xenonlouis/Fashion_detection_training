import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
from typing import List, Tuple, Dict
from rembg import remove
from PIL import Image
import io
import colorsys
from skimage.feature import graycomatrix, graycoprops
from skimage import filters

class ColorAnalyzer:
    def __init__(self, n_colors: int = 5):
        """
        Initialize the color analyzer.
        Args:
            n_colors: Number of dominant colors to extract
        """
        self.n_colors = n_colors
        # Extended color dictionary with more shades
        self.color_names = {
            # Reds
            'red': [255, 0, 0],
            'dark_red': [139, 0, 0],
            'maroon': [128, 0, 0],
            'crimson': [220, 20, 60],
            'indian_red': [205, 92, 92],
            'light_coral': [240, 128, 128],
            
            # Blues
            'blue': [0, 0, 255],
            'navy': [0, 0, 128],
            'royal_blue': [65, 105, 225],
            'steel_blue': [70, 130, 180],
            'light_blue': [173, 216, 230],
            'sky_blue': [135, 206, 235],
            'turquoise': [64, 224, 208],
            
            # Greens
            'green': [0, 255, 0],
            'dark_green': [0, 100, 0],
            'forest_green': [34, 139, 34],
            'olive': [128, 128, 0],
            'lime_green': [50, 205, 50],
            'sage_green': [138, 154, 91],
            'mint': [189, 252, 201],
            
            # Yellows
            'yellow': [255, 255, 0],
            'gold': [255, 215, 0],
            'light_yellow': [255, 255, 224],
            'khaki': [240, 230, 140],
            'mustard': [255, 219, 88],
            
            # Browns
            'brown': [165, 42, 42],
            'saddle_brown': [139, 69, 19],
            'sienna': [160, 82, 45],
            'chocolate': [210, 105, 30],
            'tan': [210, 180, 140],
            'beige': [245, 245, 220],
            
            # Purples
            'purple': [128, 0, 128],
            'violet': [238, 130, 238],
            'magenta': [255, 0, 255],
            'plum': [221, 160, 221],
            'lavender': [230, 230, 250],
            
            # Pinks
            'pink': [255, 192, 203],
            'hot_pink': [255, 105, 180],
            'deep_pink': [255, 20, 147],
            'light_pink': [255, 182, 193],
            'salmon': [250, 128, 114],
            
            # Neutrals
            'white': [255, 255, 255],
            'black': [0, 0, 0],
            'gray': [128, 128, 128],
            'silver': [192, 192, 192],
            'light_gray': [211, 211, 211],
            'dark_gray': [169, 169, 169],
            'charcoal': [54, 69, 79],
            
            # Special fashion colors
            'cream': [255, 253, 208],
            'ivory': [255, 255, 240],
            'champagne': [247, 231, 206],
            'burgundy': [128, 0, 32],
            'navy_blue': [0, 0, 128],
            'teal': [0, 128, 128],
            'coral': [255, 127, 80],
            'mauve': [224, 176, 255],
            'taupe': [72, 60, 50],
        }
        
        # Initialize color categories
        self.color_categories = {
            'warm': ['red', 'orange', 'yellow', 'brown', 'gold', 'coral', 'burgundy'],
            'cool': ['blue', 'green', 'purple', 'teal', 'turquoise', 'mint'],
            'neutral': ['black', 'white', 'gray', 'beige', 'cream', 'ivory', 'taupe'],
            'pastel': ['light_pink', 'light_blue', 'mint', 'lavender', 'cream'],
            'vibrant': ['hot_pink', 'deep_pink', 'royal_blue', 'lime_green', 'magenta']
        }
        
        # Add pattern types
        self.pattern_types = {
            'solid': 'Single dominant color with minimal variation',
            'striped': 'Regular alternating patterns',
            'patterned': 'Complex or irregular patterns',
            'gradient': 'Smooth color transitions',
            'textured': 'Regular small-scale variations'
        }

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove background from image using rembg."""
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Remove background
        img_no_bg = remove(img_pil)
        
        # Convert back to numpy array
        return cv2.cvtColor(np.array(img_no_bg), cv2.COLOR_RGBA2BGR)

    def get_color_category(self, color_name: str) -> List[str]:
        """Get the categories a color belongs to."""
        categories = []
        for category, colors in self.color_categories.items():
            if any(base_color in color_name.lower() for base_color in colors):
                categories.append(category)
        return categories if categories else ['uncategorized']

    def get_color_properties(self, rgb: List[int]) -> Dict[str, float]:
        """Get detailed color properties (HSV values)."""
        # Convert RGB to HSV
        rgb_normalized = [x/255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(*rgb_normalized)
        
        # Convert hue to degrees
        h = h * 360
        
        return {
            'hue': round(h, 2),
            'saturation': round(s * 100, 2),
            'value': round(v * 100, 2),
            'is_light': v > 0.7,
            'is_dark': v < 0.3,
            'is_saturated': s > 0.7,
            'is_muted': s < 0.3
        }

    def get_closest_color_name(self, rgb_value: np.ndarray) -> str:
        """
        Find the closest matching color name for an RGB value.
        """
        min_distance = float('inf')
        closest_color = None
        
        # Convert to HSV for better color matching
        rgb_normalized = rgb_value / 255.0
        h1, s1, v1 = colorsys.rgb_to_hsv(*rgb_normalized)
        
        for color_name, color_rgb in self.color_names.items():
            # Convert reference color to HSV
            rgb_ref_normalized = [x/255.0 for x in color_rgb]
            h2, s2, v2 = colorsys.rgb_to_hsv(*rgb_ref_normalized)
            
            # Calculate distance in HSV space (with weighted components)
            h_diff = min(abs(h1 - h2), 1 - abs(h1 - h2)) * 2.0  # Hue is circular
            s_diff = abs(s1 - s2)
            v_diff = abs(v1 - v2)
            distance = (h_diff * 2) + (s_diff * 1) + (v_diff * 1)
            
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color

    def extract_colors(self, image: np.ndarray) -> List[Dict[str, any]]:
        """
        Extract dominant colors from an image.
        Args:
            image: Input image in BGR format
        Returns:
            List of dictionaries containing color information
        """
        # Remove background
        image_no_bg = self.remove_background(image)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2RGB)
        
        # Create mask for non-transparent pixels
        mask = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY) > 0
        
        # Get only non-transparent pixels
        pixels = image_rgb[mask]
        
        if len(pixels) == 0:
            raise ValueError("No valid pixels found after background removal")
        
        # Cluster the pixels
        kmeans = KMeans(n_clusters=self.n_colors, n_init=10)
        kmeans.fit(pixels)
        
        # Get the colors and their counts
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Calculate percentages and get color information
        total_pixels = sum(counts.values())
        color_info = []
        
        for i in range(self.n_colors):
            color = colors[i]
            count = counts[i]
            percentage = count / total_pixels * 100
            
            # Get color name and properties
            color_name = self.get_closest_color_name(color)
            color_properties = self.get_color_properties(color.astype(int).tolist())
            categories = self.get_color_category(color_name)
            
            color_info.append({
                'color_name': color_name,
                'rgb': color.astype(int).tolist(),
                'percentage': round(percentage, 2),
                'properties': color_properties,
                'categories': categories
            })
        
        # Sort by percentage
        color_info.sort(key=lambda x: x['percentage'], reverse=True)
        return color_info

    def detect_pattern(self, image: np.ndarray) -> Dict[str, any]:
        """Detect the pattern type in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        
        # Calculate edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # Calculate color variation
        color_std = np.std(image.reshape(-1, 3), axis=0).mean()
        
        # Pattern classification logic
        pattern_scores = {
            'solid': 1.0 - color_std/50,  # High for uniform colors
            'striped': edge_density * (1 - homogeneity),  # High for regular edges
            'patterned': contrast * edge_density,  # High for complex patterns
            'gradient': (1 - energy) * (1 - edge_density),  # High for smooth transitions
            'textured': homogeneity * contrast  # High for regular textures
        }
        
        # Normalize scores
        total = sum(pattern_scores.values())
        if total > 0:
            pattern_scores = {k: v/total for k, v in pattern_scores.items()}
        
        # Get primary pattern type
        primary_pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate texture properties
        texture_properties = {
            'contrast': float(contrast),
            'homogeneity': float(homogeneity),
            'energy': float(energy),
            'edge_density': float(edge_density),
            'color_variation': float(color_std)
        }
        
        return {
            'primary_pattern': primary_pattern,
            'pattern_scores': pattern_scores,
            'pattern_description': self.pattern_types[primary_pattern],
            'texture_properties': texture_properties
        }

    def analyze_image_colors(self, image_path: str) -> Dict[str, any]:
        """
        Analyze colors in an image file.
        Args:
            image_path: Path to the image file
        Returns:
            Dictionary containing color analysis results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Remove background and extract colors
        image_no_bg = self.remove_background(image)
        colors = self.extract_colors(image_no_bg)
        
        # Detect pattern
        pattern_info = self.detect_pattern(image_no_bg)
        
        # Calculate color scheme
        color_scheme = {
            'warm_ratio': sum(c['percentage'] for c in colors if 'warm' in c['categories']),
            'cool_ratio': sum(c['percentage'] for c in colors if 'cool' in c['categories']),
            'neutral_ratio': sum(c['percentage'] for c in colors if 'neutral' in c['categories']),
            'is_mostly_light': any(c['properties']['is_light'] and c['percentage'] > 50 for c in colors),
            'is_mostly_dark': any(c['properties']['is_dark'] and c['percentage'] > 50 for c in colors),
            'is_high_contrast': any(c['properties']['is_light'] for c in colors) and 
                              any(c['properties']['is_dark'] for c in colors)
        }
        
        # Prepare results
        result = {
            'dominant_colors': colors,
            'primary_color': colors[0]['color_name'],
            'color_palette': [color['color_name'] for color in colors],
            'color_scheme': color_scheme,
            'color_properties': {
                color['color_name']: {
                    'percentage': color['percentage'],
                    'properties': color['properties'],
                    'categories': color['categories']
                } for color in colors
            },
            'pattern_analysis': pattern_info
        }
        
        return result 