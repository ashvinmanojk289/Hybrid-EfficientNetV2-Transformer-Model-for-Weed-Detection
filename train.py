import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import from other project files
from data_loader import load_datasets
from models import (
    create_cnn_model,
    create_transformer_model,
    create_hybrid_model,
    create_efficientnet_transformer_model
)

# --- Configuration ---
PREPARED_DATASET_DIR = 'prepared_dataset'
NUM_EPOCHS = 50
NUM_RUNS = 5
RESULTS_DIR = 'training_results'

def plot_history(history, model_name, run):
    """Saves plots of training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title(f'{model_name} - Run {run} - Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax1.grid(True)

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f'{model_name} - Run {run} - Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(RESULTS_DIR, model_name, f'run_{run}_history.png')
    plt.savefig(plot_filename)
    plt.close()

def train_model(model_fn, model_name, train_ds, val_ds):
    """Trains a given model for NUM_RUNS and saves results."""
    print("\n" + "="*60)
    print(f"Starting Training for: {model_name}")
    print("="*60)

    # Create directories for this model's results
    model_results_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    
    run_histories = []

    for i in range(1, NUM_RUNS + 1):
        print(f"\n--- Run {i}/{NUM_RUNS} for {model_name} ---")
        
        # Create a new instance of the model for each run
        model = model_fn()
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Set up callbacks
        model_checkpoint_path = os.path.join(model_results_dir, f'best_model_run_{i}.keras')
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10, # Stop if val_accuracy doesn't improve for 10 epochs
            restore_best_weights=True
        )

        # Train the model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=NUM_EPOCHS,
            callbacks=[checkpoint_cb, early_stopping_cb],
            verbose=1
        )

        # Save history plot
        plot_history(history, model_name, i)

        # Store history data
        hist_df = pd.DataFrame(history.history)
        hist_df['run'] = i
        run_histories.append(hist_df)
    
    # Combine and save all run histories to a single CSV
    all_runs_df = pd.concat(run_histories, ignore_index=True)
    csv_path = os.path.join(model_results_dir, 'all_runs_history.csv')
    all_runs_df.to_csv(csv_path, index=False)
    print(f"\nSaved training history for all runs to {csv_path}")


def main():
    """Main function to run the entire training pipeline."""
    # Create main results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    try:
        train_ds, val_ds, _, _ = load_datasets(PREPARED_DATASET_DIR)
    except FileNotFoundError:
        print(f"Error: The directory '{PREPARED_DATASET_DIR}' was not found.")
        print("Please run 'prepare_data.py' first to create it.")
        return

    # Define all models to be trained
    models_to_train = {
        "Standard_CNN": create_cnn_model,
        "Vision_Transformer": create_transformer_model,
        "Hybrid_CNN_Transformer": create_hybrid_model,
        "EfficientNetV2B0_Transformer": create_efficientnet_transformer_model,
    }

    # Train each model
    start_time = datetime.now()
    for name, model_function in models_to_train.items():
        train_model(model_function, name, train_ds, val_ds)
    
    end_time = datetime.now()
    print("\n" + "*"*60)
    print("All training complete!")
    print(f"Total training time: {end_time - start_time}")
    print("*"*60)


if __name__ == '__main__':
    main()
