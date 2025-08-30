import tensorflow as tf

# --- Configuration ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

def load_datasets(prepared_data_dir):
    """
    Loads and preprocesses the training, validation, and test datasets.

    Args:
        prepared_data_dir (str): Path to the directory created by prepare_data.py.

    Returns:
        tuple: A tuple containing the train, validation, and test tf.data.Dataset objects.
    """
    print("Loading and preprocessing datasets...")

    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'{prepared_data_dir}/train',
        labels='inferred',
        label_mode='int',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Load validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f'{prepared_data_dir}/valid',
        labels='inferred',
        label_mode='int',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Load test data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f'{prepared_data_dir}/test',
        labels='inferred',
        label_mode='int',
        image_size=IMG_SIZE,
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")

    # --- Preprocessing ---
    # Define a normalization layer
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    def preprocess_dataset(ds):
        # Apply normalization and configure for performance
        ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = preprocess_dataset(train_ds)
    val_ds = preprocess_dataset(val_ds)
    test_ds = preprocess_dataset(test_ds)

    print("Datasets loaded and preprocessed successfully.")
    return train_ds, val_ds, test_ds, class_names

if __name__ == '__main__':
    # This is an example of how to use the function.
    # This script is intended to be imported, not run directly.
    PREPARED_DATASET_DIR = 'prepared_dataset'
    try:
        train_dataset, val_dataset, test_dataset, classes = load_datasets(PREPARED_DATASET_DIR)
        print("\n--- Dataset Specs ---")
        print("Train Dataset:", train_dataset)
        print("Validation Dataset:", val_dataset)
        print("Test Dataset:", test_dataset)
        print("Classes:", classes)
        # Print one batch shape
        for images, labels in train_dataset.take(1):
            print("Image batch shape:", images.shape)
            print("Label batch shape:", labels.shape)
    except FileNotFoundError:
        print(f"\nError: The directory '{PREPARED_DATASET_DIR}' was not found.")
        print("Please run 'prepare_data.py' first to create it.")
