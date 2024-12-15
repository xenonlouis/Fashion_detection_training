import tensorflow as tf
from keras import layers, Model

class FashionClassifier(Model):
    def __init__(self, num_categories, num_attributes):
        super(FashionClassifier, self).__init__()
        
        print(f"\nModel Configuration:")
        print(f"Number of categories: {num_categories}")
        print(f"Number of attributes: {num_attributes}")
        
        # Base model
        self.base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        self.base_model.trainable = False
        
        # Global average pooling
        self.global_avg = layers.GlobalAveragePooling2D()
        
        # Common dense layers
        self.dense1 = layers.Dense(512, 
                                 kernel_initializer='glorot_uniform',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.bn1 = layers.BatchNormalization(epsilon=1e-5)
        self.act1 = layers.ReLU()
        self.dropout = layers.Dropout(0.2)
        
        # Category specific layers
        self.category_dense = layers.Dense(256, 
                                         kernel_initializer='glorot_uniform',
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.category_bn = layers.BatchNormalization(epsilon=1e-5)
        self.category_act = layers.ReLU()
        self.category_dropout = layers.Dropout(0.2)
        self.category_output = layers.Dense(
            num_categories, 
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            activation=None,  # Remove activation here
            name='output_1'
        )
        self.category_softmax = layers.Activation('softmax', dtype='float32')
        
        # Attribute specific layers
        self.attribute_dense = layers.Dense(256, 
                                          kernel_initializer='glorot_uniform',
                                          kernel_regularizer=tf.keras.regularizers.l2(1e-5))
        self.attribute_bn = layers.BatchNormalization(epsilon=1e-5)
        self.attribute_act = layers.ReLU()
        self.attribute_dropout = layers.Dropout(0.2)
        self.attribute_output = layers.Dense(
            num_attributes, 
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            activation=None,  # Remove activation here
            name='output_2'
        )
        self.attribute_sigmoid = layers.Activation('sigmoid', dtype='float32')
    
    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_avg(x)
        
        # Common layers
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dropout(x, training=training)
        
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