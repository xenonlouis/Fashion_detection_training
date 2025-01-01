import tensorflow as tf
from keras import layers, Model

class FashionClassifier(Model):
    def __init__(self, num_categories, num_attributes):
        super(FashionClassifier, self).__init__()
        
        # Define regularizer
        regularizer = tf.keras.regularizers.l2(1e-5)
        
        # Memory-efficient backbone for RTX 3060 Mobile
        self.base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Enable gradient checkpointing to save memory
        self.base_model.trainable = True
        
        # Simplified architecture with smaller dimensions
        self.global_avg = layers.GlobalAveragePooling2D()
        
        # Common feature extraction with reduced dimensions
        self.dense1 = layers.Dense(512, 
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizer)
        self.bn1 = layers.BatchNormalization(epsilon=1e-6)
        self.act1 = layers.LeakyReLU(alpha=0.1)
        self.dropout1 = layers.Dropout(0.3)
        
        # Category branch with reduced dimensions
        self.category_dense = layers.Dense(256, 
                                         kernel_initializer='he_normal',
                                         kernel_regularizer=regularizer)
        self.category_bn = layers.BatchNormalization(epsilon=1e-6)
        self.category_act = layers.LeakyReLU(alpha=0.1)
        self.category_dropout = layers.Dropout(0.3)
        self.category_output = layers.Dense(num_categories, name='output_1')
        self.category_softmax = layers.Activation('softmax', dtype='float32')
        
        # Attribute branch with reduced dimensions
        self.attribute_dense = layers.Dense(256, 
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=regularizer)
        self.attribute_bn = layers.BatchNormalization(epsilon=1e-6)
        self.attribute_act = layers.LeakyReLU(alpha=0.1)
        self.attribute_dropout = layers.Dropout(0.3)
        self.attribute_output = layers.Dense(num_attributes, name='output_2')
        self.attribute_sigmoid = layers.Activation('sigmoid', dtype='float32')
    
    def call(self, inputs, training=False):
        # Forward pass with memory optimizations
        x = self.base_model(inputs, training=training)
        x = self.global_avg(x)
        
        # Common features
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout1(x, training=training)
        
        # Category branch
        category_x = self.category_dense(x)
        category_x = self.category_bn(category_x, training=training)
        category_x = self.category_act(category_x)
        category_x = self.category_dropout(category_x, training=training)
        category_x = self.category_output(category_x)
        category_output = self.category_softmax(category_x)
        
        # Attribute branch
        attribute_x = self.attribute_dense(x)
        attribute_x = self.attribute_bn(attribute_x, training=training)
        attribute_x = self.attribute_act(attribute_x)
        attribute_x = self.attribute_dropout(attribute_x, training=training)
        attribute_x = self.attribute_output(attribute_x)
        attribute_output = self.attribute_sigmoid(attribute_x)
        
        return category_output, attribute_output