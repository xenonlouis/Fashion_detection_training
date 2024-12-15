import tensorflow as tf
import numpy as np
from model.fashion_classifier import FashionClassifier
from keras import mixed_precision
import os
import pandas as pd

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_image(image_path):
    # Convert tensor to string and handle path joining using tf operations
    base_path = "./deepfashion/img/"
    full_path = tf.strings.join([base_path, image_path])
    
    image = tf.io.read_file(full_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

def create_dataset(image_paths, category_labels, attribute_labels):
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, (category_labels, attribute_labels))
    )
    dataset = dataset.map(
        lambda x, y: (preprocess_image(x), y),
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

def create_weighted_sparse_categorical_crossentropy(class_weights):
    def weighted_loss(y_true, y_pred):
        # Convert class weights to tensor
        weights_tensor = tf.convert_to_tensor(
            [class_weights[i] for i in range(len(class_weights))], 
            dtype=tf.float32
        )
        
        # Get weights for each sample based on their true class
        sample_weights = tf.gather(weights_tensor, tf.cast(y_true, tf.int32))
        
        # Calculate regular sparse categorical crossentropy
        unweighted_losses = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        
        # Apply weights to the losses
        weighted_losses = unweighted_losses * sample_weights
        
        return tf.reduce_mean(weighted_losses)
    
    return weighted_loss

# Add this new class after the create_weighted_sparse_categorical_crossentropy function
class AttributeSpecificMetrics(tf.keras.callbacks.Callback):
    def __init__(self, attribute_names):
        super(AttributeSpecificMetrics, self).__init__()
        self.attribute_names = attribute_names
        
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for the entire dataset
        predictions = self.model.predict(self.model.train_dataset)
        y_pred = predictions[1] > 0.5  # Binary predictions for attributes
        y_true = np.concatenate([y[1] for x, y in self.model.train_dataset])
        
        print("\nPer-attribute metrics:")
        for i, attr_name in enumerate(self.attribute_names):
            accuracy = np.mean(y_pred[:, i] == y_true[:, i])
            precision = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1)) / (np.sum(y_pred[:, i] == 1) + 1e-7)
            recall = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1)) / (np.sum(y_true[:, i] == 1) + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            
            print(f"{attr_name:15s}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

def load_deepfashion_dataset():
    # Define file paths
    anno_dir = "./deepfashion/Anno/"
    
    # Load attribute and category lists
    attr_cloth_file = os.path.join(anno_dir, "list_attr_cloth.txt")
    attr_img_file = os.path.join(anno_dir, "list_attr_img.txt")
    
    # Read attribute types - first read the count
    with open(attr_cloth_file, 'r') as f:
        attr_count = int(f.readline().strip())
        # Skip the header line
        next(f)
        # Now read the attributes with pandas
        attr_types = pd.read_csv(f, delimiter='\s+', 
                               names=['attribute_name', 'attribute_type'])
    
    # Read image attributes - first read the count
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
    
    # Verify the first image path
    print(f"Sample image path: {image_paths[0]}")
    print(f"Full image path: {os.path.join('./deepfashion', image_paths[0])}")
    
    # Load attributes and categories with additional checks
    train_attributes = pd.read_csv(train_attr_file, delimiter='\s+', 
                                 header=None, dtype=np.float32).values
    train_categories = pd.read_csv(train_cate_file, header=None, 
                                 dtype=np.int32).values.ravel()
    
    # Enhanced data validation
    print("\nDetailed Data Validation:")
    print(f"Categories range: {train_categories.min()} to {train_categories.max()}")
    print(f"Unique categories: {np.unique(train_categories)}")
    print(f"Category counts: {np.bincount(train_categories)}")
    
    print(f"\nAttributes stats:")
    print(f"Min value: {train_attributes.min()}")
    print(f"Max value: {train_attributes.max()}")
    print(f"Mean value: {train_attributes.mean()}")
    print(f"Sample of first 5 attribute vectors:")
    print(train_attributes[:5])
    
    # Distribution of categories
    unique_cats, cat_counts = np.unique(train_categories, return_counts=True)
    print("\nCategory distribution:")
    for cat, count in zip(unique_cats, cat_counts):
        print(f"Category {cat}: {count} samples")
    
    # Load category names and total number of categories
    with open(os.path.join(anno_dir, "list_category_cloth.txt"), 'r') as f:
        num_total_categories = int(f.readline().strip())  # Should be 50
        next(f)  # Skip header
        category_names = pd.read_csv(f, delimiter='\s+', 
                                   names=['category_name', 'category_type'])
    
    # Verify categories are 1-based index and go up to 50
    print(f"\nTotal categories in list_category_cloth: {num_total_categories}")
    print(f"Current categories in training data: {len(np.unique(train_categories))}")
    
    # No need to remap categories - keep original indices
    # Just verify all categories are within valid range
    assert train_categories.min() >= 1 and train_categories.max() <= num_total_categories, \
        f"Categories should be between 1 and {num_total_categories}"
    
    # First adjust categories to be 0-based
    train_categories = train_categories - 1
    
    # Calculate class weights for all possible categories (0 to 49)
    total_samples = len(train_categories)
    class_weights = {}
    
    # Initialize weights for all possible categories
    for cat in range(num_total_categories):
        samples = np.sum(train_categories == cat)
        if samples == 0:
            # For categories that don't appear in training set
            class_weights[cat] = 1.0
        else:
            # Calculate weight based on inverse frequency
            class_weights[cat] = total_samples / (len(np.unique(train_categories)) * samples)
    
    print("\nClass weights (first 5):")
    for i in range(5):
        print(f"Category {i}: {class_weights[i]:.2f}")
    
    # Verify weights are properly 0-based and complete
    min_cat = min(class_weights.keys())
    max_cat = max(class_weights.keys())
    num_weights = len(class_weights)
    print(f"\nClass weight range: {min_cat} to {max_cat}")
    print(f"Number of class weights: {num_weights}")
    assert num_weights == num_total_categories, f"Expected {num_total_categories} weights, got {num_weights}"
    
    print(f"Loaded {len(image_paths)} training images")
    print(f"Number of attributes: {len(attr_types)}")
    print(f"Attribute names: {attr_types['attribute_name'].tolist()[:5]}...")
    print(f"Sample image path: {image_paths[0]}")
    print(f"Sample attributes shape: {train_attributes.shape}")
    
    # Get attribute names for metrics
    attribute_names = attr_types['attribute_name'].tolist()
    
    return np.array(image_paths), train_categories, train_attributes, class_weights, num_total_categories, attribute_names

def main():
    # Add at the beginning of main()
    mixed_precision.set_global_policy('mixed_float16')
    
    # Load your DeepFashion dataset
    train_images, train_categories, train_attributes, class_weights, num_total_categories, attribute_names = load_deepfashion_dataset()
    
    # Create model with total number of categories (50)
    model = FashionClassifier(num_total_categories, train_attributes.shape[1])
    
    num_categories = len(np.unique(train_categories))
    num_attributes = train_attributes.shape[1]
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=1.0,
        epsilon=1e-8,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Create weighted loss function
    weighted_category_loss = create_weighted_sparse_categorical_crossentropy(class_weights)
    
    # Create learning rate scheduler first
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    # Then create callbacks list
    callbacks = [
        lr_schedule,
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/model_{epoch:02d}',
            save_best_only=True,
            monitor='loss',
            save_format='tf'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger('training_log.csv'),
        AttributeSpecificMetrics(attribute_names)
    ]
    
    # Add per-attribute metrics to model compilation
    attribute_metrics = [
        [
            tf.keras.metrics.BinaryAccuracy(name=f'attr_{i}_{name}_accuracy'),
            tf.keras.metrics.Precision(name=f'attr_{i}_{name}_precision'),
            tf.keras.metrics.Recall(name=f'attr_{i}_{name}_recall')
        ]
        for i, name in enumerate(attribute_names)
    ]
    
    # Flatten the list of metrics
    all_attribute_metrics = [metric for metrics_list in attribute_metrics for metric in metrics_list]
    
    model.compile(
        optimizer=optimizer,
        loss={
            'output_1': weighted_category_loss,
            'output_2': tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
                reduction='auto'
            )
        },
        loss_weights={
            'output_1': 1.0,
            'output_2': 1.0
        },
        metrics={
            'output_1': [
                'accuracy',
                tf.keras.metrics.SparseCategoricalAccuracy(name='top1_accuracy'),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')
            ],
            'output_2': [
                'binary_accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ] + all_attribute_metrics  # Add flattened attribute metrics
        }
    )
    
    # Create dataset
    train_dataset = create_dataset(train_images, train_categories, train_attributes)
    
    # Train with more detailed metrics
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main() 