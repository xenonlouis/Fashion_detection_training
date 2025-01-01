import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
import logging
import traceback
from datetime import datetime
from model.fashion_classifier import FashionClassifier
from keras import mixed_precision
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = 224
AUTOTUNE = tf.data.AUTOTUNE

# Set mixed precision policy once
mixed_precision.set_global_policy('mixed_float16')
print('Compute dtype: %s' % mixed_precision.global_policy().compute_dtype)
print('Variable dtype: %s' % mixed_precision.global_policy().variable_dtype)

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/test_run_{timestamp}.log'
    
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

def preprocess_image(image_path):
    # Convert tensor to string and handle path joining using tf operations
    base_path = "./deepfashion/img/"
    full_path = tf.strings.join([base_path, image_path])
    
    image = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    
    # EfficientNetV2L normalization
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image = (image - mean) / std
    
    return image

def create_dataset(image_paths, category_labels, attribute_labels):
    # Convert sparse labels to one-hot
    category_labels_onehot = tf.one_hot(category_labels, depth=50)  # 50 categories
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, (category_labels_onehot, attribute_labels))
    )
    dataset = dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

def load_test_dataset():
    # Define file paths
    anno_dir = "./deepfashion/Anno/"
    
    # Load test data paths
    test_img_paths_file = os.path.join(anno_dir, "test.txt")
    test_attr_file = os.path.join(anno_dir, "test_attr.txt")
    test_cate_file = os.path.join(anno_dir, "test_cate.txt")
    
    # Get attribute names
    attr_cloth_file = os.path.join(anno_dir, "list_attr_cloth.txt")
    with open(attr_cloth_file, 'r') as f:
        num_attributes = int(f.readline().strip())
        next(f)  # Skip header
        attr_types = pd.read_csv(f, delimiter='\s+', 
                               names=['attribute_name', 'attribute_type'])
    
    # Read test data
    test_images = pd.read_csv(test_img_paths_file, header=None, delimiter=' ')[0].tolist()
    test_attributes = pd.read_csv(test_attr_file, delimiter='\s+', 
                                header=None, dtype=np.float32).values
    test_categories = pd.read_csv(test_cate_file, header=None, 
                                dtype=np.int32).values.ravel()
    
    # Adjust categories to be 0-based
    test_categories = test_categories - 1
    
    logging.info(f"\nTest Data:")
    logging.info(f"Number of test images: {len(test_images)}")
    logging.info(f"Test categories shape: {test_categories.shape}")
    logging.info(f"Test attributes shape: {test_attributes.shape}")
    
    return np.array(test_images), test_categories, test_attributes, attr_types['attribute_name'].tolist()

def plot_performance_graphs(category_accuracy, top5_accuracy, attribute_metrics, attribute_names):
    # Set style
    plt.style.use('default')  # Use default style instead of seaborn
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Category Classification Performance
    plt.subplot(2, 2, 1)
    accuracies = [category_accuracy, top5_accuracy]
    plt.bar(['Top-1 Accuracy', 'Top-5 Accuracy'], accuracies, color='skyblue')
    plt.title('Category Classification Performance', pad=20)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 2. Attribute Performance Overview
    plt.subplot(2, 2, 2)
    metrics_mean = {
        'Accuracy': np.mean([m['accuracy'] for m in attribute_metrics]),
        'Precision': np.mean([m['precision'] for m in attribute_metrics]),
        'Recall': np.mean([m['recall'] for m in attribute_metrics]),
        'F1': np.mean([m['f1'] for m in attribute_metrics])
    }
    plt.bar(metrics_mean.keys(), metrics_mean.values(), color='lightgreen')
    plt.title('Average Attribute Performance', pad=20)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 3. Top 10 Best Performing Attributes
    plt.subplot(2, 1, 2)
    df = pd.DataFrame(attribute_metrics)
    df['attribute'] = attribute_names
    df = df.sort_values('f1', ascending=True).tail(10)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(df))
    plt.barh(y_pos, df['f1'], color='salmon')
    plt.yticks(y_pos, df['attribute'])
    plt.xlabel('F1 Score')
    plt.title('Top 10 Best Performing Attributes', pad=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed attribute performance heatmap
    plt.figure(figsize=(15, len(attribute_names)//2))
    data = pd.DataFrame(attribute_metrics)
    data.index = attribute_names
    
    # Create heatmap with custom colors
    sns.heatmap(data[['accuracy', 'precision', 'recall', 'f1']], 
                annot=True, fmt='.3f', 
                cmap='RdYlGn',  # Red-Yellow-Green colormap
                center=0.5,     # Center the colormap at 0.5
                vmin=0, vmax=1)
    plt.title('Detailed Attribute Performance', pad=20)
    plt.tight_layout()
    plt.savefig('attribute_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_dataset, attribute_names):
    try:
        # Evaluate the model
        logging.info("\nEvaluating model on test set...")
        logging.info("Running model.evaluate()...")
        results = model.evaluate(test_dataset, verbose=1)
        logging.info(f"Evaluation results: {results}")
        
        logging.info("\nRunning predictions...")
        predictions = model.predict(test_dataset, verbose=1)
        logging.info("Predictions completed")
        
        if not isinstance(predictions, (list, tuple)) or len(predictions) != 2:
            logging.warning(f"Warning: Unexpected predictions format. Got: {type(predictions)}")
            if isinstance(predictions, (list, tuple)):
                logging.warning(f"Predictions length: {len(predictions)}")
            return
        
        category_preds, attribute_preds = predictions
        logging.info(f"Category predictions shape: {category_preds.shape}")
        logging.info(f"Attribute predictions shape: {attribute_preds.shape}")
        
        # Get true labels from the dataset
        y_true_cat_onehot = []
        y_true_attr = []
        for _, y in test_dataset:
            y_true_cat_onehot.append(y[0])
            y_true_attr.append(y[1])
        
        y_true_cat_onehot = np.concatenate(y_true_cat_onehot)
        y_true_attr = np.concatenate(y_true_attr)
        
        # Convert one-hot back to sparse for metrics
        y_true_cat = np.argmax(y_true_cat_onehot, axis=1)
        y_pred_cat = np.argmax(category_preds, axis=1)
        
        # Clip attribute predictions to prevent numerical instability
        attribute_preds = np.clip(attribute_preds, -100, 100)
        y_pred_attr = (tf.nn.sigmoid(attribute_preds) > 0.5).numpy().astype(int)
        
        # Overall metrics
        category_accuracy = np.mean(y_true_cat == y_pred_cat)
        top5_accuracy = results[model.metrics_names.index('output_1_top5_accuracy')]
        logging.info(f"\nOverall Results:")
        logging.info(f"Category Accuracy: {category_accuracy:.4f}")
        logging.info(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        
        # Per-attribute metrics with numerical stability
        logging.info("\nPer-attribute Results:")
        attribute_metrics = []
        for i, attr_name in enumerate(attribute_names):
            # Use epsilon to prevent division by zero
            epsilon = 1e-7
            
            true_positives = np.sum((y_pred_attr[:, i] == 1) & (y_true_attr[:, i] == 1))
            false_positives = np.sum((y_pred_attr[:, i] == 1) & (y_true_attr[:, i] == 0))
            false_negatives = np.sum((y_pred_attr[:, i] == 0) & (y_true_attr[:, i] == 1))
            
            accuracy = np.mean(y_pred_attr[:, i] == y_true_attr[:, i])
            precision = true_positives / (true_positives + false_positives + epsilon)
            recall = true_positives / (true_positives + false_negatives + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
            # Clip metrics to valid range [0, 1]
            precision = np.clip(precision, 0, 1)
            recall = np.clip(recall, 0, 1)
            f1 = np.clip(f1, 0, 1)
            
            attribute_metrics.append({
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            })
            
            logging.info(f"{attr_name:30s}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
        
        # Plot performance graphs
        plot_performance_graphs(category_accuracy, top5_accuracy, attribute_metrics, attribute_names)
        logging.info("\nPerformance graphs have been saved as 'performance_summary.png' and 'attribute_performance_heatmap.png'")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        logging.error(traceback.format_exc())

def main():
    try:
        # Setup logging
        setup_logging()
        
        # Load test data
        logging.info("\nStep 1: Loading test data...")
        test_images, test_categories, test_attributes, attribute_names = load_test_dataset()
        logging.info("[OK] Test data loaded successfully")
        
        # Create test dataset
        logging.info("\nStep 2: Creating test dataset...")
        test_dataset = create_dataset(test_images, test_categories, test_attributes)
        logging.info("[OK] Test dataset created successfully")
        
        # Create model with same architecture
        logging.info("\nStep 3: Creating model...")
        num_total_categories = 50  # From list_category_cloth.txt
        num_attributes = 26        # From list_attr_cloth.txt
        model = FashionClassifier(num_total_categories, num_attributes)
        logging.info("[OK] Model created successfully")
        
        # Create a properly preprocessed dummy input to build the model
        logging.info("\nStep 4: Building model architecture...")
        dummy_input = tf.random.uniform((1, IMAGE_SIZE, IMAGE_SIZE, 3), minval=0, maxval=255, dtype=tf.float32)
        dummy_input = tf.cast(dummy_input, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        dummy_input = (dummy_input - mean) / std
        with tf.device('/CPU:0'):  # Force CPU to prevent any GPU memory issues
            _ = model(dummy_input, training=False)
        logging.info("[OK] Model architecture built successfully")
        
        # Set up optimizer and compile model
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
        
        # Load the weights
        logging.info("\nStep 5: Loading model weights...")
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
        
        # Print model summary
        logging.info("\nModel Summary:")
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        for line in model_summary:
            logging.info(line)
        
        # Evaluate the model
        logging.info("\nStep 6: Evaluating model...")
        evaluate_model(model, test_dataset, attribute_names)
        logging.info("[OK] Evaluation completed successfully")
        
    except Exception as e:
        logging.error(f"\nError occurred: {str(e)}")
        import traceback
        logging.error("\nFull traceback:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 