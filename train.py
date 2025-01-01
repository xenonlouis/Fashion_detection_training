import tensorflow as tf
import numpy as np
from model.fashion_classifier import FashionClassifier
from keras import mixed_precision
import os
import pandas as pd
from tensorflow.keras.callbacks import LearningRateScheduler

# GPU Memory Growth setting
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f'Found {len(physical_devices)} GPU(s)')
else:
    print('No GPU found, using CPU')

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 3
AUTOTUNE = tf.data.AUTOTUNE
VALIDATION_SPLIT = 0.2
SHUFFLE_BUFFER = 10000  # Increased shuffle buffer

# Set mixed precision policy once
mixed_precision.set_global_policy('mixed_float16')
print('Compute dtype: %s' % mixed_precision.global_policy().compute_dtype)
print('Variable dtype: %s' % mixed_precision.global_policy().variable_dtype)

def cosine_decay_with_warmup(epoch, total_epochs, warmup_epochs=3, learning_rate_base=1e-4, warmup_learning_rate=0.0):
    if epoch < warmup_epochs:
        return warmup_learning_rate + (learning_rate_base - warmup_learning_rate) * (epoch / warmup_epochs)
    
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return learning_rate_base * 0.5 * (1 + np.cos(np.pi * progress))

def preprocess_image(image_path):
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

@tf.function
def augment_image(image):
    """Basic image augmentations."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image

@tf.function
def mixup_batch(images, labels_cat, labels_attr, alpha=0.2):
    """Performs MixUp on batches of images and labels."""
    batch_size = tf.shape(images)[0]
    
    # Create random weights for mixing
    weights = tf.random.uniform([batch_size], dtype=tf.float32)
    x_weights = tf.reshape(weights, [batch_size, 1, 1, 1])
    cat_weights = tf.reshape(weights, [batch_size, 1])
    attr_weights = tf.reshape(weights, [batch_size, 1])
    
    # Create shuffled indices
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Mix the data
    mixed_images = (
        images * x_weights + 
        tf.gather(images, indices) * (1 - x_weights)
    )
    
    # Mix the labels (already in one-hot format for categories)
    mixed_labels_cat = (
        labels_cat * cat_weights + 
        tf.gather(labels_cat, indices) * (1 - cat_weights)
    )
    mixed_labels_attr = (
        labels_attr * attr_weights + 
        tf.gather(labels_attr, indices) * (1 - attr_weights)
    )
    
    return mixed_images, (mixed_labels_cat, mixed_labels_attr)

def create_dataset(image_paths, category_labels, attribute_labels, is_training=True):
    # Create the initial dataset
    # Convert sparse labels to one-hot
    category_labels_onehot = tf.one_hot(category_labels, depth=50)  # 50 categories
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, (category_labels_onehot, attribute_labels))
    )
    
    if is_training:
        # Cache the raw data
        dataset = dataset.cache()
        # Shuffle before preprocessing
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    
    # Preprocess images
    dataset = dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=AUTOTUNE
    )
    
    if is_training:
        # Apply augmentations before batching
        dataset = dataset.map(
            lambda x, y: (augment_image(x), y),
            num_parallel_calls=AUTOTUNE
        )
    
    # Batch the dataset
    dataset = dataset.batch(BATCH_SIZE)
    
    if is_training:
        # Apply mixup after batching
        dataset = dataset.map(
            lambda x, y: mixup_batch(x, y[0], y[1]),
            num_parallel_calls=AUTOTUNE
        )
    
    # Prefetch for performance
    return dataset.prefetch(AUTOTUNE)

class AttributeSpecificMetrics(tf.keras.callbacks.Callback):
    def __init__(self, attribute_names, validation_data=None):
        super(AttributeSpecificMetrics, self).__init__()
        self.attribute_names = attribute_names
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        # Get validation dataset
        val_dataset = self.validation_data
        if val_dataset is None and hasattr(self.model, 'validation_data'):
            val_dataset = self.model.validation_data
        
        if val_dataset is None:
            print("Warning: No validation data found for attribute metrics")
            return
        
        # Get predictions in batches to avoid memory issues
        y_pred_list = []
        y_true_list = []
        
        try:
            # Iterate over validation dataset
            for x_batch, y_batch in val_dataset.take(-1):  # Take all batches
                pred_batch = self.model.predict_on_batch(x_batch)
                if isinstance(pred_batch, (list, tuple)):
                    pred_batch = pred_batch[1]  # Get attribute predictions
                if isinstance(y_batch, (list, tuple)):
                    y_batch = y_batch[1]  # Get attribute labels
                    
                y_pred_list.append(pred_batch)
                y_true_list.append(y_batch)
            
            # Concatenate all batches
            if y_pred_list and y_true_list:
                y_pred = np.concatenate(y_pred_list) > 0.5
                y_true = np.concatenate(y_true_list)
                
                # Update metrics
                for i, attr_name in enumerate(self.attribute_names):
                    accuracy = np.mean(y_pred[:, i] == y_true[:, i])
                    precision = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1)) / (np.sum(y_pred[:, i] == 1) + 1e-7)
                    recall = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1)) / (np.sum(y_true[:, i] == 1) + 1e-7)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
                    
                    logs[f'attr_{i}_accuracy'] = accuracy
                    logs[f'attr_{i}_precision'] = precision
                    logs[f'attr_{i}_recall'] = recall
                    logs[f'attr_{i}_f1'] = f1
        except Exception as e:
            print(f"Warning: Error computing attribute metrics: {str(e)}")

def load_deepfashion_dataset():
    # Define file paths
    anno_dir = "./deepfashion/Anno/"
    
    # Load attribute and category lists
    attr_cloth_file = os.path.join(anno_dir, "list_attr_cloth.txt")
    attr_img_file = os.path.join(anno_dir, "list_attr_img.txt")
    category_cloth_file = os.path.join(anno_dir, "list_category_cloth.txt")
    
    # Read attribute types
    with open(attr_cloth_file, 'r') as f:
        attr_count = int(f.readline().strip())
        # Skip the header line
        next(f)
        # Now read the attributes with pandas
        attr_types = pd.read_csv(f, delimiter='\s+', 
                               names=['attribute_name', 'attribute_type'])
    
    # Read image attributes
    with open(attr_img_file, 'r') as f:
        img_count = int(f.readline().strip())
        # Skip the header line
        next(f)
        # Now read the attributes with pandas
        attr_img = pd.read_csv(f, delimiter='\s+',
                             names=['image_name'] + [f'attr_{i}' for i in range(26)])
    
    # Load training data paths
    train_img_paths_file = os.path.join(anno_dir, "train.txt")
    train_attr_file = os.path.join(anno_dir, "train_attr.txt")
    train_cate_file = os.path.join(anno_dir, "train_cate.txt")
    
    # Read image paths
    image_paths = pd.read_csv(train_img_paths_file, header=None, delimiter=' ')[0].tolist()
    
    # Load attributes and categories
    train_attributes = pd.read_csv(train_attr_file, delimiter='\s+', 
                                 header=None, dtype=np.float32).values
    train_categories = pd.read_csv(train_cate_file, header=None, 
                                 dtype=np.int32).values.ravel()
    
    # Load category names and total number of categories
    with open(category_cloth_file, 'r') as f:
        num_total_categories = int(f.readline().strip())
        next(f)  # Skip header
        category_names = pd.read_csv(f, delimiter='\s+', 
                                   names=['category_name', 'category_type'])
    
    # Calculate class weights
    total_samples = len(train_categories)
    class_weights = {}
    unique_cats, cat_counts = np.unique(train_categories, return_counts=True)
    
    # Initialize weights for all possible categories (1-based to 0-based indexing)
    for cat in range(num_total_categories):
        cat_idx = cat + 1  # Convert to 1-based indexing to match data
        count = cat_counts[unique_cats == cat_idx][0] if cat_idx in unique_cats else 0
        if count == 0:
            class_weights[cat] = 1.0
        else:
            class_weights[cat] = total_samples / (num_total_categories * count)
    
    # Convert categories to 0-based indexing
    train_categories = train_categories - 1
    
    # Data validation
    print("\nDataset Statistics:")
    print(f"Number of training images: {len(image_paths)}")
    print(f"Number of attributes: {train_attributes.shape[1]}")
    print(f"Number of categories: {num_total_categories}")
    print(f"\nCategory distribution:")
    for cat, count in zip(unique_cats, cat_counts):
        print(f"Category {cat}: {count} samples")
    
    # Get attribute names for metrics
    attribute_names = attr_types['attribute_name'].tolist()
    
    return np.array(image_paths), train_categories, train_attributes, class_weights, num_total_categories, attribute_names

def main():
    # GPU check
    if not tf.test.is_built_with_cuda():
        print("WARNING: TensorFlow was not built with CUDA support")
    else:
        print("TensorFlow was built with CUDA support")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Cuda version: {tf.sysconfig.get_build_info()['cuda_version']}")
        print(f"CUDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    
    # Load dataset
    train_images, train_categories, train_attributes, class_weights, num_total_categories, attribute_names = load_deepfashion_dataset()
    
    # Split data for validation
    num_samples = len(train_images)
    num_val = int(num_samples * VALIDATION_SPLIT)
    indices = np.random.permutation(num_samples)
    train_idx, val_idx = indices[num_val:], indices[:num_val]
    
    # Create datasets with optimized pipeline
    train_dataset = create_dataset(
        train_images[train_idx], 
        train_categories[train_idx], 
        train_attributes[train_idx],
        is_training=True
    )
    
    val_dataset = create_dataset(
        train_images[val_idx], 
        train_categories[val_idx], 
        train_attributes[val_idx],
        is_training=False
    )
    
    # Create model (no need to specify mixed precision again)
    model = FashionClassifier(num_total_categories, train_attributes.shape[1])
    
    # Optimizer setup
    base_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        epsilon=1e-8,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
    
    # Learning rate schedule
    lr_schedule = LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(epoch, EPOCHS)
    )
    
    callbacks = [
        lr_schedule,
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/model_weights_{epoch:02d}-{val_loss:.2f}',
            save_best_only=True,
            monitor='val_loss',
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger('training_log.csv', separator=',', append=False),
        AttributeSpecificMetrics(attribute_names, validation_data=val_dataset),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
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
    
    # Train with updated settings
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
        validation_freq=1  # Ensure validation runs every epoch
    )

if __name__ == "__main__":
    main() 