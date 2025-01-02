from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import uvicorn
import logging
from datetime import datetime
import os
from model.fashion_classifier import FashionClassifier
from model.response_mapper import map_prediction_to_frontend
from model.color_analyzer import ColorAnalyzer
from typing import Optional, List
from tensorflow.keras import mixed_precision
import traceback
import base64

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/api_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Constants
IMAGE_SIZE = 224
NUM_CATEGORIES = 50
NUM_ATTRIBUTES = 26

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')
logging.info('Compute dtype: %s' % mixed_precision.global_policy().compute_dtype)
logging.info('Variable dtype: %s' % mixed_precision.global_policy().variable_dtype)

# Initialize FastAPI app
app = FastAPI(
    title="Fashion Analysis API",
    description="API for analyzing fashion items in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and analyzer
model = None
color_analyzer = None
category_names = []
attribute_names = []

def load_category_and_attribute_names():
    """Load category and attribute names from files."""
    global category_names, attribute_names
    
    anno_dir = "./deepfashion/Anno/"
    
    # Load category names
    with open(os.path.join(anno_dir, "list_category_cloth.txt"), 'r') as f:
        num_categories = int(f.readline().strip())
        next(f)  # Skip header
        categories = []
        for line in f:
            category = line.strip().split()[0]
            categories.append(category)
    
    # Load attribute names
    with open(os.path.join(anno_dir, "list_attr_cloth.txt"), 'r') as f:
        num_attributes = int(f.readline().strip())
        next(f)  # Skip header
        attributes = []
        for line in f:
            attribute = line.strip().split()[0]
            attributes.append(attribute)
    
    category_names = categories
    attribute_names = attributes

def preprocess_image(image: Image.Image) -> tf.Tensor:
    """Preprocess image for model prediction."""
    # Resize image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to array and normalize
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.cast(image_array, tf.float32) / 255.0
    
    # Apply normalization
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image_array = (image_array - mean) / std
    
    # Convert to mixed precision if needed
    if mixed_precision.global_policy().compute_dtype == 'float16':
        image_array = tf.cast(image_array, tf.float16)
    
    # Add batch dimension
    return tf.expand_dims(image_array, 0)

@app.on_event("startup")
async def startup_event():
    """Initialize model and analyzer on startup."""
    global model, color_analyzer
    
    try:
        # Load names
        load_category_and_attribute_names()
        
        # Initialize model with mixed precision
        with tf.device('/CPU:0'):  # Initialize on CPU to avoid memory issues
            model = FashionClassifier(NUM_CATEGORIES, NUM_ATTRIBUTES)
            
            # Create optimizer with mixed precision
            base_optimizer = tf.keras.optimizers.Adam(
                learning_rate=1e-4,
                epsilon=1e-8,
                beta_1=0.9,
                beta_2=0.999
            )
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss={
                    'output_1': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'output_2': tf.keras.losses.BinaryCrossentropy(from_logits=False)
                }
            )
        
        # Load weights
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.index')]
        if not checkpoints:
            raise ValueError("No checkpoint files found")
        
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getctime(os.path.join('checkpoints', x)))
        checkpoint_path = os.path.join('checkpoints', latest_checkpoint[:-6])
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load weights with partial state restoration
        load_status = model.load_weights(checkpoint_path).expect_partial()
        logging.info("Model loaded successfully")
        
        # Initialize color analyzer
        color_analyzer = ColorAnalyzer()
        logging.info("Color analyzer initialized")
        
    except Exception as e:
        logging.error(f"Error during startup: {str(e)}")
        logging.error(traceback.format_exc())
        raise


@app.get("/")
async def root():
    return {"message": "Welcome to the Fashion Analysis API"}


@app.post("/predict")
async def predict_image(request: Request):
    """
    Predict fashion attributes from base64 image data.
    """
    try:
        logging.info("Processing prediction request")
        
        # Get JSON data from request
        data = await request.json()
        logging.info("Received image data for prediction")
        
        image_base64 = data.get('image')
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        try:
            # Convert base64 to image
            image_data = base64.b64decode(image_base64.split(',')[1] if ',' in image_base64 else image_base64)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logging.error(f"Error decoding image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        processed_image = preprocess_image(image)
        logging.info("Image preprocessed successfully")
        
        # Get predictions
        predictions = model(processed_image, training=False)
        logging.info("Model predictions completed")
        
        if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
            logging.error(f"Unexpected predictions format: {type(predictions)}")
            raise ValueError(f"Unexpected predictions format")
        
        category_preds, attribute_preds = predictions
        
        # Ensure predictions are not empty
        if len(category_preds) == 0 or len(category_preds[0]) == 0:
            logging.error("Empty category predictions")
            raise ValueError("Empty predictions")
            
        # Format initial results
        results = {
            'top_categories': [
                {
                    'category': category_names[idx],
                    'confidence': float(prob)
                }
                for idx, prob in enumerate(category_preds[0])
                if float(prob) > 0.1
            ]
        }
        
        if len(results['top_categories']) == 0:
            logging.warning("No categories above confidence threshold")
            results['top_categories'] = [{
                'category': category_names[0],
                'confidence': float(category_preds[0][0])
            }]
        
        # Add attributes
        results['attributes'] = [
            {
                'attribute': attr_name,
                'confidence': float(prob)
            }
            for attr_name, prob in zip(attribute_names, tf.nn.sigmoid(attribute_preds[0]))
            if float(prob) > 0.5
        ]
        
        logging.info(f"Predictions processed: {len(results['top_categories'])} categories, {len(results['attributes'])} attributes")
        
        # Add color analysis
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image.save(temp_path)
        
        try:
            color_results = color_analyzer.analyze_image_colors(temp_path)
            results.update(color_results)
            logging.info("Color analysis completed")
        except Exception as e:
            logging.error(f"Error in color analysis: {str(e)}")
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Map results to frontend format
        frontend_response = map_prediction_to_frontend(results)
        logging.info("Response mapped to frontend format")
        
        return frontend_response
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their processing time."""
    start_time = datetime.now()
    
    # Log request details
    logging.info(f"Request started: {request.method} {request.url.path}")
    logging.info(f"Client IP: {request.client.host}")
    logging.info(f"Headers: {dict(request.headers)}")
    
    # Process request
    response = await call_next(request)
    
    # Log response details
    process_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Request completed: {request.method} {request.url.path}")
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Processing Time: {process_time:.2f}s")
    logging.info("-" * 50)
    
    return response

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 