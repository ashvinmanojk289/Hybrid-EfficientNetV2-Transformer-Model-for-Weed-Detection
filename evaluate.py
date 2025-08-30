import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_datasets

# --- Configuration ---
PREPARED_DATASET_DIR = 'prepared_dataset'
RESULTS_DIR = 'training_results'
MODELS_TO_EVALUATE = [
    "Standard_CNN",
    "Vision_Transformer",
    "Hybrid_CNN_Transformer",
    "EfficientNetV2B0_Transformer",
]
NUM_RUNS = 5

def plot_confusion_matrix(cm, class_names, model_name, run):
    """Saves a confusion matrix plot."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Run {run} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Ensure directory exists
    model_eval_dir = os.path.join(RESULTS_DIR, model_name, 'evaluation')
    os.makedirs(model_eval_dir, exist_ok=True)

    plot_path = os.path.join(model_eval_dir, f'run_{run}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()


def evaluate_models():
    """Loads trained models and evaluates them on the test set."""
    print("Starting model evaluation...")
    
    # Load test dataset
    try:
        _, _, test_ds, class_names = load_datasets(PREPARED_DATASET_DIR)
    except FileNotFoundError:
        print(f"Error: The directory '{PREPARED_DATASET_DIR}' was not found.")
        print("Please run 'prepare_data.py' and 'train.py' first.")
        return
        
    # Get true labels from the test dataset
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    all_results = []

    for model_name in MODELS_TO_EVALUATE:
        print("\n" + "="*60)
        print(f"Evaluating: {model_name}")
        print("="*60)
        
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if not os.path.exists(model_dir):
            print(f"Warning: Results directory for {model_name} not found. Skipping.")
            continue

        for i in range(1, NUM_RUNS + 1):
            print(f"--- Evaluating Run {i}/{NUM_RUNS} ---")
            
            model_path = os.path.join(model_dir, f'best_model_run_{i}.keras')
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found for run {i} at {model_path}. Skipping.")
                continue

            # Load the best model from the run
            model = tf.keras.models.load_model(model_path)
            
            # Evaluate loss and accuracy
            loss, accuracy = model.evaluate(test_ds, verbose=0)
            print(f"  Test Loss: {loss:.4f}")
            print(f"  Test Accuracy: {accuracy:.4f}")
            
            # Generate predictions
            y_pred_probs = model.predict(test_ds, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Classification Report
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            print("  Classification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, class_names, model_name, i)
            
            # Store results
            result = {
                'model': model_name,
                'run': i,
                'accuracy': accuracy,
                'loss': loss,
                'precision_macro': report['macro avg']['precision'],
                'recall_macro': report['macro avg']['recall'],
                'f1_score_macro': report['macro avg']['f1-score'],
            }
            all_results.append(result)

    # Save all evaluation metrics to a single CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        eval_csv_path = os.path.join(RESULTS_DIR, 'evaluation_summary.csv')
        results_df.to_csv(eval_csv_path, index=False)
        print(f"\nSaved comprehensive evaluation summary to {eval_csv_path}")

        # Display summary statistics
        summary = results_df.groupby('model')['accuracy', 'f1_score_macro'].agg(['mean', 'std'])
        print("\n" + "*"*60)
        print("Average Performance Across All Runs")
        print("*"*60)
        print(summary)

if __name__ == '__main__':
    evaluate_models()
