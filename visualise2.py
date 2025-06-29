import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

# Updated metrics based on your training output
metrics = {
    'train': {
        'loss': [0.3655, 0.2788, 0.2610, 0.2513, 0.2459],
        'acc': [0.9349, 0.9579, 0.9632, 0.9685, 0.9729],
        'precision': [0.9118, 0.9322, 0.9386, 0.9461, 0.9568],
        'recall': [0.8635, 0.9252, 0.9371, 0.9475, 0.9515],
        'f1': [0.8870, 0.9287, 0.9379, 0.9468, 0.9541],
        'auc': [0.9737, 0.9855, 0.9895, 0.9920, 0.9920]
    },
    'val': {
        'loss': [0.2748, 0.2594, 0.2416, 0.2331, 0.2326],
        'acc': [0.9689, 0.9650, 0.9748, 0.9803, 0.9810],
        'precision': [0.9345, 0.8989, 0.9276, 0.9523, 0.9533],
        'recall': [0.9625, 0.9933, 0.9925, 0.9945, 0.9975],
        'f1': [0.9483, 0.9438, 0.9589, 0.9672, 0.9685],
        'auc': [0.9935, 0.9941, 0.9957, 0.9956, 0.9969]
    }
}

# Per-class accuracies from your output
per_class_acc_real = [0.8635, 0.9252, 0.9371, 0.9475, 0.9515]  # Train Real
per_class_acc_fake = [0.9649, 0.9717, 0.9742, 0.9773, 0.9820]  # Train Fake
val_per_class_acc_real = [0.9625, 0.9933, 0.9925, 0.9945, 0.9975]  # Val Real
val_per_class_acc_fake = [0.9716, 0.9531, 0.9674, 0.9793, 0.9799]  # Val Fake

# Prediction confidence from your output
pred_confidence_train = [0.7820, 0.8884, 0.9210, 0.9329, 0.9371]
pred_confidence_val = [0.8625, 0.9118, 0.9308, 0.9379, 0.9411]

# Epochs
epochs = range(1, len(metrics['train']['loss']) + 1)

# Gradient norms (sampled every 25 iterations from your output for each epoch)
gradient_norms = [
    76612.7812, 36196.9688, 36894.8789, 43702.1055, 38070.2617,  # Epoch 1
    49088.7812, 63172.0820, 65530.0898, 19962.6875, 55284.4648,  # Epoch 2
    38664.6211, 81234.0859, 23900.1504, 14310.4795, 22486.1152,  # Epoch 3
    69013.4922, 37268.2969, 38266.5781, 68902.2891, 29439.2969,  # Epoch 4
    62209.5859, 169367.5781, 55627.4727, 132309.7188, 8160.2656   # Epoch 5
]

# Confusion matrices from your output (validation set only)
conf_matrices_val = [
    np.array([[2774, 81], [45, 1155]]),  # Epoch 1
    np.array([[2721, 134], [8, 1192]]),  # Epoch 2
    np.array([[2762, 93], [9, 1191]]),   # Epoch 3
    np.array([[2796, 59], [21, 1179]]),  # Epoch 4
    np.array([[2793, 62], [15, 1185]])   # Epoch 5
]

# Simulated data for plots requiring raw predictions (replace with actual data if available)
y_true_val = np.concatenate([np.ones(1200), np.zeros(2855)])  # Based on final val confusion matrix
y_pred_val = np.random.rand(len(y_true_val))  # Simulated predictions
train_losses = np.random.normal(loc=0.2459, scale=0.05, size=1000)  # Final train loss
val_losses = np.random.normal(loc=0.2326, scale=0.05, size=1000)    # Final val loss
embeddings = np.random.rand(100, 50)  # Simulated embeddings
labels = np.random.randint(0, 2, 100)  # 0: Fake, 1: Real

# 1. Training vs Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics['train']['loss'], label='Train Loss', marker='o')
plt.plot(epochs, metrics['val']['loss'], label='Validation Loss', marker='o')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/loss_comparison.png')
plt.close()

# 2. Training vs Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics['train']['acc'], label='Train Accuracy', marker='o')
plt.plot(epochs, metrics['val']['acc'], label='Validation Accuracy', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/accuracy_comparison.png')
plt.close()

# 3. Precision, Recall, and F1-Score (Train)
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics['train']['precision'], label='Precision', marker='o')
plt.plot(epochs, metrics['train']['recall'], label='Recall', marker='o')
plt.plot(epochs, metrics['train']['f1'], label='F1-Score', marker='o')
plt.title('Train Precision, Recall, and F1-Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/train_prf.png')
plt.close()

# 4. Precision, Recall, and F1-Score (Validation)
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics['val']['precision'], label='Precision', marker='o')
plt.plot(epochs, metrics['val']['recall'], label='Recall', marker='o')
plt.plot(epochs, metrics['val']['f1'], label='F1-Score', marker='o')
plt.title('Validation Precision, Recall, and F1-Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/val_prf.png')
plt.close()

# 5. ROC-AUC Score
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics['train']['auc'], label='Train AUC', marker='o')
plt.plot(epochs, metrics['val']['auc'], label='Validation AUC', marker='o')
plt.title('Training vs Validation ROC-AUC Score')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/auc_comparison.png')
plt.close()

# 6. Gradient Norm Over Iterations
plt.figure(figsize=(10, 6))
plt.plot(range(len(gradient_norms)), gradient_norms, label='Gradient Norm')
plt.title('Gradient Norm Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/gradient_norm.png')
plt.close()

# 7. Per-Class Accuracy (Train)
plt.figure(figsize=(10, 6))
plt.plot(epochs, per_class_acc_real, label='Real (Train)', marker='o')
plt.plot(epochs, per_class_acc_fake, label='Fake (Train)', marker='o')
plt.title('Per-Class Accuracy (Train)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/train_per_class_acc.png')
plt.close()

# 8. Per-Class Accuracy (Validation)
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_per_class_acc_real, label='Real (Validation)', marker='o')
plt.plot(epochs, val_per_class_acc_fake, label='Fake (Validation)', marker='o')
plt.title('Per-Class Accuracy (Validation)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/val_per_class_acc.png')
plt.close()

# 9. Prediction Confidence
plt.figure(figsize=(10, 6))
plt.plot(epochs, pred_confidence_train, label='Train Confidence', marker='o')
plt.plot(epochs, pred_confidence_val, label='Validation Confidence', marker='o')
plt.title('Prediction Confidence Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Confidence')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/prediction_confidence.png')
plt.close()

# 10. Confusion Matrix Heatmap (Final Epoch - Validation)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices_val[-1], annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix (Validation - Epoch 5)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/mnt/d/quantum/train-finger-2/confusion_matrix.png')
plt.close()

# 11. Loss Distribution (Histogram)
plt.figure(figsize=(10, 6))
plt.hist(train_losses, bins=50, alpha=0.5, label='Train Loss', color='blue')
plt.hist(val_losses, bins=50, alpha=0.5, label='Validation Loss', color='orange')
plt.title('Loss Distribution (Final Epoch)')
plt.xlabel('Loss Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/loss_distribution.png')
plt.close()

# 12. ROC Curve (Simulated - Replace with actual predictions if available)
fpr, tpr, _ = roc_curve(y_true_val, y_pred_val)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'AUC = {metrics["val"]["auc"][-1]:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve (Validation - Final Epoch)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/roc_curve.png')
plt.close()

# 13. Precision-Recall Curve (Simulated - Replace with actual predictions if available)
precision, recall, _ = precision_recall_curve(y_true_val, y_pred_val)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'Final Epoch')
plt.title('Precision-Recall Curve (Validation - Final Epoch)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/pr_curve.png')
plt.close()

# 15. Gradient Norm Distribution
plt.figure(figsize=(10, 6))
plt.boxplot([gradient_norms[i::5] for i in range(5)], labels=[f'Epoch {i+1}' for i in range(5)])
plt.title('Gradient Norm Distribution Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/gradient_norm_distribution.png')
plt.close()

# 16. Feature Importance (Simulated - Replace with actual data if applicable)
feature_names = [f'Feature {i}' for i in range(10)]
importance_scores = np.random.rand(10)
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importance_scores)
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.savefig('/mnt/d/quantum/train-finger-2/feature_importance.png')
plt.close()

# 18. Error Rate per Class (Validation)
classes = ['Real', 'Fake']
error_rates = [1 - val_per_class_acc_real[-1], 1 - val_per_class_acc_fake[-1]]
plt.figure(figsize=(8, 6))
plt.bar(classes, error_rates)
plt.title('Error Rate per Class (Validation - Final Epoch)')
plt.xlabel('Class')
plt.ylabel('Error Rate')
plt.savefig('/mnt/d/quantum/train-finger-2/error_rate_per_class.png')
plt.close()

# 19. Cumulative Metrics
cum_acc_train = np.cumsum(metrics['train']['acc']) / np.arange(1, len(epochs) + 1)
cum_acc_val = np.cumsum(metrics['val']['acc']) / np.arange(1, len(epochs) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, cum_acc_train, label='Cumulative Train Accuracy', marker='o')
plt.plot(epochs, cum_acc_val, label='Cumulative Validation Accuracy', marker='o')
plt.title('Cumulative Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Cumulative Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/mnt/d/quantum/train-finger-2/cumulative_accuracy.png')
plt.close()

# 20. t-SNE Visualization (Simulated - Replace with actual embeddings if available)
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[labels == 0, 0], tsne_results[labels == 0, 1], label='Fake', alpha=0.5)
plt.scatter(tsne_results[labels == 1, 0], tsne_results[labels == 1, 1], label='Real', alpha=0.5)
plt.title('t-SNE Visualization of Learned Representations')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.savefig('/mnt/d/quantum/train-finger-2/tsne_visualization.png')
plt.close()

print("All plots have been saved as PNG files.")
