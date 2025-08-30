import tensorflow as tf
from tensorflow.keras import layers, models, applications

# --- Configuration ---
IMG_SIZE = (256, 256)
NUM_CLASSES = 2  # Weed, Crop

# ==============================================================================
# Model 1: Standard Convolutional Neural Network (CNN)
# ==============================================================================
def create_cnn_model():
    """Builds a standard CNN model."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name="Standard_CNN")
    return model

# ==============================================================================
# Model 2: Vision Transformer (ViT)
# ==============================================================================
def create_transformer_model(
    num_heads=4, projection_dim=64, transformer_layers=4, mlp_dim=128
):
    """Builds a Vision Transformer model."""
    patch_size = 16
    num_patches = (IMG_SIZE[0] // patch_size) ** 2
    transformer_units = [projection_dim * 2, projection_dim]

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Create patches
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="VALID",
        name="patch_creation"
    )(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    # Create positional embedding
    position_embedding = layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(tf.range(num_patches))
    encoded_patches = patches + position_embedding

    # Create Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        for units in transformer_units:
            x3 = layers.Dense(units, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Final classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    features = layers.Dense(mlp_dim, activation='relu')(representation)
    logits = layers.Dense(NUM_CLASSES, activation='softmax')(features)

    model = models.Model(inputs=inputs, outputs=logits, name="Vision_Transformer")
    return model

# ==============================================================================
# Model 3: Hybrid CNN-Transformer
# ==============================================================================
def create_hybrid_model():
    """Builds a Hybrid CNN-Transformer model."""
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # CNN Feature Extractor
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    feature_maps = layers.MaxPooling2D((2, 2))(x)
    
    # Reshape feature maps into patches for the Transformer
    _, h, w, c = feature_maps.shape
    patches = layers.Reshape((h * w, c))(feature_maps)
    
    # Transformer Block
    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(patches)
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=c, dropout=0.1
    )(encoded_patches, encoded_patches)
    x2 = layers.Add()([attention_output, encoded_patches])
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = layers.Dense(c * 2, activation=tf.nn.gelu)(x3)
    x3 = layers.Dense(c, activation=tf.nn.gelu)(x3)
    encoded_patches = layers.Add()([x3, x2])
    
    # Classification Head
    x = layers.GlobalAveragePooling1D()(encoded_patches)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_Transformer")
    return model

# ==============================================================================
# Model 4: EfficientNetV2B0 + Transformer
# ==============================================================================
def create_efficientnet_transformer_model():
    """Builds an EfficientNetV2B0 base with a Transformer head."""
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Load pre-trained EfficientNetV2B0
    base_model = applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    base_model.trainable = False # Freeze base model initially

    # Get the output feature map
    x = base_model.output
    
    # Reshape for Transformer
    _, h, w, c = x.shape
    patches = layers.Reshape((h * w, c))(x)

    # Transformer Block
    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(patches)
    attention_output = layers.MultiHeadAttention(
        num_heads=8, key_dim=c, dropout=0.1
    )(encoded_patches, encoded_patches)
    
    # Classification Head
    x = layers.GlobalAveragePooling1D()(attention_output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="EfficientNetV2B0_Transformer")
    return model


if __name__ == '__main__':
    # Print summaries of all models to verify their architecture
    print("\n" + "="*50)
    print("Model 1: Standard CNN")
    print("="*50)
    cnn_model = create_cnn_model()
    cnn_model.summary()

    print("\n" + "="*50)
    print("Model 2: Vision Transformer")
    print("="*50)
    transformer_model = create_transformer_model()
    transformer_model.summary()

    print("\n" + "="*50)
    print("Model 3: Hybrid CNN-Transformer")
    print("="*50)
    hybrid_model = create_hybrid_model()
    hybrid_model.summary()
    
    print("\n" + "="*50)
    print("Model 4: EfficientNetV2B0 + Transformer")
    print("="*50)
    effnet_transformer_model = create_efficientnet_transformer_model()
    effnet_transformer_model.summary()
