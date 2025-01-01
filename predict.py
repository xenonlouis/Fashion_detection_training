import tensorflow as tf
import numpy as np
import pandas as pd
import os
from model.fashion_classifier import FashionClassifier
import glob
from keras import mixed_precision
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec
import logging
from datetime import datetime
import traceback
from model.color_analyzer import ColorAnalyzer
from typing import Dict

# Constants
IMAGE_SIZE = 224

# Initialize color analyzer
color_analyzer = ColorAnalyzer(n_colors=5)

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/predict_run_{timestamp}.log'
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f'Log file created at: {log_file}')

# Initialize logging first
setup_logging()

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')
logging.info('Compute dtype: %s' % mixed_precision.global_policy().compute_dtype)
logging.info('Variable dtype: %s' % mixed_precision.global_policy().variable_dtype)

# Set style for modern look
plt.style.use('default')  # Use default style instead of seaborn
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc'
})

def load_category_and_attribute_names():
    anno_dir = "./deepfashion/Anno/"
    
    # Load category names
    with open(os.path.join(anno_dir, "list_category_cloth.txt"), 'r') as f:
        num_categories = int(f.readline().strip())
        next(f)  # Skip header
        categories = pd.read_csv(f, delimiter='\s+', 
                               names=['category_name', 'category_type'])
    
    # Load attribute names
    with open(os.path.join(anno_dir, "list_attr_cloth.txt"), 'r') as f:
        num_attributes = int(f.readline().strip())
        next(f)  # Skip header
        attributes = pd.read_csv(f, delimiter='\s+', 
                               names=['attribute_name', 'attribute_type'])
    
    return categories['category_name'].tolist(), attributes['attribute_name'].tolist()

def preprocess_image(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Read and preprocess image using PIL first
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to tensor
        image = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0
        
        # EfficientNetV2L normalization
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - mean) / std
        
        # Add batch dimension
        image = tf.expand_dims(image, 0)
        return image
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")

def predict_fashion(model, image_path, category_names, attribute_names):
    # Preprocess image
    image = preprocess_image(image_path)
    
    # Get predictions
    predictions = model(image, training=False)
    if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
        raise ValueError(f"Unexpected predictions format. Got: {type(predictions)}")
    
    category_preds, attribute_preds = predictions
    
    # Get top 5 categories
    top5_indices = tf.argsort(category_preds[0], direction='DESCENDING')[:5]
    top5_probs = tf.gather(category_preds[0], top5_indices)
    
    # Convert attribute logits to probabilities
    attribute_probs = tf.nn.sigmoid(attribute_preds[0])
    
    # Format base results
    results = {
        'image_path': image_path,
        'top_categories': [
            {
                'category': category_names[idx],
                'confidence': float(prob)
            }
            for idx, prob in zip(top5_indices.numpy(), top5_probs.numpy())
        ],
        'attributes': [
            {
                'attribute': attr_name,
                'confidence': float(prob)
            }
            for attr_name, prob in zip(attribute_names, attribute_probs.numpy())
            if float(prob) > 0.5  # Only include attributes with >50% confidence
        ]
    }
    
    # Analyze colors
    try:
        color_results = color_analyzer.analyze_image_colors(image_path)
        if color_results:
            # Convert color results to the expected format
            formatted_colors = []
            for color in color_results.get('dominant_colors', []):
                formatted_colors.append({
                    'name': color.get('name', 'Unknown'),
                    'percentage': color.get('percentage', 0),
                    'rgb': color.get('rgb', [0, 0, 0])
                })
            
            results['colors'] = formatted_colors
            results['primary_color'] = color_results.get('primary_color', {}).get('name', 'Unknown')
            results['color_scheme'] = {
                'temperature': color_results.get('color_scheme', {}).get('temperature', 'neutral'),
                'brightness': color_results.get('color_scheme', {}).get('brightness', 'medium')
            }
    except Exception as e:
        logging.warning(f"Color analysis failed: {str(e)}")
    
    return results

def visualize_prediction(image_path, results):
    # Create figure with custom layout
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[2, 1])
    
    # Top left subplot for original image
    ax_img = plt.subplot(gs[0, 0])
    img = Image.open(image_path)
    ax_img.imshow(img)
    ax_img.set_title('Original Image', pad=10)
    ax_img.axis('off')
    
    # Top right subplot for predictions and color swatches
    ax_pred = plt.subplot(gs[0, 1])
    ax_pred.axis('off')
    
    # Create color swatches if color data exists
    if 'colors' in results and results['colors']:
        n_colors = len(results['colors'])
        swatch_height = 0.8 / max(n_colors, 1)
        
        for i, color in enumerate(results['colors']):
            if color.get('percentage', 0) > 0.05:  # Only show colors with >5% presence
                # Create color rectangle
                rgb = np.array(color.get('rgb', [0, 0, 0])) / 255.0
                rect = plt.Rectangle((0.7, 0.9 - (i+1)*swatch_height), 0.2, swatch_height,
                                   facecolor=rgb, edgecolor='black', linewidth=1)
                ax_pred.add_patch(rect)
                
                # Add color percentage
                ax_pred.text(0.92, 0.9 - (i+0.5)*swatch_height, 
                            f"{color.get('percentage', 0):.1%}",
                            ha='left', va='center')
    
    # Add predictions text
    text_content = []
    
    # Add title
    text_content.append("Fashion Analysis Results")
    text_content.append("-" * 40)
    
    # Add categories
    text_content.append("\nTop Categories:")
    for i, cat in enumerate(results['top_categories'], 1):
        confidence = cat['confidence'] * 100
        if confidence > 20:  # Only show categories with >20% confidence
            text_content.append(f"{i}. {cat['category']} ({confidence:.1f}%)")
    
    # Add attributes
    if results['attributes']:
        text_content.append("\nDetected Attributes:")
        sorted_attrs = sorted(results['attributes'], 
                            key=lambda x: x['confidence'], 
                            reverse=True)
        for attr in sorted_attrs:
            confidence = attr['confidence'] * 100
            text_content.append(f"• {attr['attribute']} ({confidence:.1f}%)")
    
    # Join all text with newlines
    full_text = '\n'.join(text_content)
    
    # Add text to plot
    ax_pred.text(0, 1, full_text, 
                fontsize=11, 
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(facecolor='white', 
                         edgecolor='lightgray',
                         boxstyle='round,pad=1'))
    
    # Bottom subplot for color analysis
    ax_color = plt.subplot(gs[1, :])
    ax_color.axis('off')
    
    # Add color analysis text if color data exists
    color_text = []
    color_text.append("\nColor Analysis")
    color_text.append("-" * 40)
    
    if 'colors' in results and results['colors']:
        # Add primary color if available
        if 'primary_color' in results:
            primary_color = str(results['primary_color']).replace('_', ' ').title()
            color_text.append(f"\nPrimary Color: {primary_color}")
        
        # Add color scheme information if available
        if 'color_scheme' in results:
            scheme = results['color_scheme']
            color_text.append("\nColor Properties:")
            if 'temperature' in scheme:
                color_text.append(f"• Temperature: {scheme['temperature']}")
            if 'brightness' in scheme:
                color_text.append(f"• Brightness: {scheme['brightness']}")
        
        # Add color distribution
        color_text.append("\nColor Distribution:")
        for color in results['colors']:
            if color.get('percentage', 0) > 0.05:  # Only show colors with >5% presence
                name = color.get('name', 'Unknown')
                percentage = color.get('percentage', 0)
                color_text.append(f"• {name}: {percentage:.1%}")
    else:
        color_text.append("\nNo color analysis available")
    
    # Join color text with newlines
    full_color_text = '\n'.join(color_text)
    
    # Add color text to plot
    ax_color.text(0, 1, full_color_text, 
                 fontsize=11, 
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(facecolor='white', 
                          edgecolor='lightgray',
                          boxstyle='round,pad=1'))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = 'predictions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save with filename based on input
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f'prediction_{base_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def print_results(results):
    print(f"\nResults for {os.path.basename(results['image_path'])}:")
    print("\nTop 5 Categories:")
    for i, cat in enumerate(results['top_categories'], 1):
        print(f"{i}. {cat['category']} ({cat['confidence']:.1%} confidence)")
    
    print("\nDetected Attributes:")
    for attr in results['attributes']:
        print(f"- {attr['attribute']} ({attr['confidence']:.1%} confidence)")
    
    if 'colors' in results:
        print("\nColor Analysis:")
        if 'primary_color' in results:
            print(f"Primary Color: {results['primary_color']}")
        print("\nColor Distribution:")
        for color_info in results['colors']:
            if isinstance(color_info, dict):
                name = color_info.get('name', 'Unknown')
                percentage = color_info.get('percentage', 0)
                print(f"- {name}: {percentage:.1%}")
        
        if 'color_scheme' in results:
            print("\nColor Properties:")
            scheme = results['color_scheme']
            if 'temperature' in scheme:
                print(f"Temperature: {scheme['temperature']}")
            if 'brightness' in scheme:
                print(f"Brightness: {scheme['brightness']}")

def main():
    try:
        # Setup logging
        setup_logging()
        
        # Load category and attribute names
        category_names, attribute_names = load_category_and_attribute_names()
        
        # Create model
        logging.info("\nStep 1: Creating model...")
        model = FashionClassifier(len(category_names), len(attribute_names))
        logging.info("[OK] Model created successfully")
        
        # Create a properly preprocessed dummy input to build the model
        logging.info("\nStep 2: Building model architecture...")
        dummy_input = tf.random.uniform((1, IMAGE_SIZE, IMAGE_SIZE, 3), minval=0, maxval=255, dtype=tf.float32)
        dummy_input = tf.cast(dummy_input, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        dummy_input = (dummy_input - mean) / std
        with tf.device('/CPU:0'):  # Force CPU to prevent any GPU memory issues
            _ = model(dummy_input, training=False)
        logging.info("[OK] Model architecture built successfully")
        
        # Set up optimizer and compile model with same settings
        logging.info("\nStep 3: Compiling model...")
        base_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            epsilon=1e-8,
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=1.0
        )
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        
        # Compile model with loss scaling for mixed precision
        model.compile(
            optimizer=optimizer,
            loss={
                'output_1': tf.keras.losses.CategoricalCrossentropy(
                    from_logits=False,
                    label_smoothing=0.1
                ),
                'output_2': tf.keras.losses.BinaryCrossentropy(
                    from_logits=False
                )
            },
            loss_weights={
                'output_1': 2.0,
                'output_2': 0.5
            },
            metrics={
                'output_1': [
                    'accuracy',
                    tf.keras.metrics.CategoricalAccuracy(name='top1_accuracy'),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
                ],
                'output_2': [
                    'binary_accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            }
        )
        logging.info("[OK] Model compiled successfully")
        
        # Load the weights
        logging.info("\nStep 4: Loading model weights...")
        checkpoints = glob.glob('checkpoints/model_weights_*')
        if not checkpoints:
            raise ValueError("No checkpoint found in ./checkpoints directory")
        
        # Filter out non-.index files and get the base checkpoint path
        index_checkpoints = [cp for cp in checkpoints if cp.endswith('.index')]
        if not index_checkpoints:
            raise ValueError("No valid checkpoint files found")
        
        latest_checkpoint = max(index_checkpoints, key=os.path.getctime)
        base_checkpoint = latest_checkpoint[:-6]  # Remove .index
        logging.info(f"Found checkpoint: {base_checkpoint}")
        
        # Load weights with verbose output and handle partial loading
        try:
            logging.info("Attempting to load weights...")
            load_status = model.load_weights(base_checkpoint).expect_partial()
            
            # Log matched and unmatched objects
            logging.info("\nWeight loading details:")
            try:
                # Try to assert all existing objects matched
                load_status.assert_existing_objects_matched()
                logging.info("[OK] All existing model variables matched with checkpoint")
            except Exception as e:
                logging.warning(f"Some variables did not match: {str(e)}")
            
            # Log any missing or unexpected keys
            if hasattr(load_status, 'missing_keys') and load_status.missing_keys:
                logging.warning("\nMissing keys (present in model but not in checkpoint):")
                for key in load_status.missing_keys:
                    logging.warning(f"  - {key}")
            
            if hasattr(load_status, 'unexpected_keys') and load_status.unexpected_keys:
                logging.warning("\nUnexpected keys (present in checkpoint but not in model):")
                for key in load_status.unexpected_keys:
                    logging.warning(f"  - {key}")
            
            logging.info("\n[OK] Weights loaded successfully (with partial loading)")
            
        except Exception as e:
            logging.error(f"Error loading weights: {str(e)}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            raise
        
        # Print model summary to log
        logging.info("\nModel Summary:")
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        for line in model_summary:
            logging.info(line)
        
        # Example usage with a list of image paths
        test_images = [
            '../test_img/t1.jpg',
            '../test_img/t2.jpg',
            '../test_img/t3.jpeg',
            'deepfashion/img/img/Sweet_Crochet_Blouse/img_00000070.jpg',
            'deepfashion/img/img/Boxy_Wide_Neck_Tee/img_00000001.jpg'
        ]
        
        logging.info("\nProcessing images...")
        for image_path in test_images:
            try:
                # Normalize path and check if file exists
                image_path = os.path.normpath(image_path)
                if not os.path.exists(image_path):
                    logging.warning(f"Image not found: {image_path}")
                    continue
                
                results = predict_fashion(model, image_path, category_names, attribute_names)
                output_path = visualize_prediction(image_path, results)
                logging.info(f"Saved prediction visualization to: {output_path}")
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                logging.error(traceback.format_exc())
        
        logging.info("\nAll predictions have been saved to the 'predictions' directory.")
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 