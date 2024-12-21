import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setting parameters
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 150
NUM_CLASSES = 2
LEARNING_RATE = 0.0001

# Helper functions
def load_dataset(csv_path, image_dir):
    data = pd.read_csv(csv_path)
    images, labels = [], []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        img_path = os.path.join(image_dir, row['filename'])
        image = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        image = tf.keras.utils.img_to_array(image) / 255.0
        label = [row['xmin'] / row['width'], row['ymin'] / row['height'], row['xmax'] / row['width'], row['ymax'] / row['height']]
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Define YOLO-like model
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation='sigmoid')(x)  # Bounding box regression
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Load dataset
train_images, train_labels = load_dataset('train/_annotations.csv', 'train')
val_images, val_labels = load_dataset('valid/_annotations.csv', 'valid')
test_images, test_labels = load_dataset('test/_annotations.csv', 'test')

# Build and compile model
model = build_model()
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
csv_logger = CSVLogger('training_log.csv')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, csv_logger]
)

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save predictions
predictions = model.predict(test_images)
for idx, pred in enumerate(predictions[:16]):
    img = test_images[idx]
    xmin, ymin, xmax, ymax = pred[:4] * IMG_SIZE
    plt.figure()
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='red', facecolor='none', lw=2))
    plt.savefig(f'result_{idx}.png')
    plt.close()

# Calculate IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0

# Evaluate IoU thresholds
ious = []
iou_threshold = 0.5
for true_box, pred_box in zip(test_labels, predictions):
    iou = calculate_iou(true_box, pred_box)
    ious.append(iou)

ious = np.array(ious)
precision = np.sum(ious >= iou_threshold) / len(ious) if len(ious) > 0 else 0
recall = np.sum(ious >= iou_threshold) / len(test_labels) if len(test_labels) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Calculate metrics vs confidence
confidence_thresholds = np.linspace(0, 1, 50)  # Adjust the number of thresholds as needed
precision_values = []
recall_values = []
f1_values = []

for threshold in confidence_thresholds:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_box, pred_box in zip(test_labels, predictions):
        iou = calculate_iou(true_box, pred_box)
        if iou >= threshold:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(test_labels) - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)

# Plot metrics
def plot_metric_vs_confidence(confidences, metrics, label, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(confidences, metrics, label=label, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_metric_vs_confidence(confidence_thresholds, precision_values, 'Precision', 'Precision vs Confidence', 'Confidence', 'Precision')
plot_metric_vs_confidence(confidence_thresholds, recall_values, 'Recall', 'Recall vs Confidence', 'Confidence', 'Recall')
plot_metric_vs_confidence(confidence_thresholds, f1_values, 'F1-Score', 'F1-Score vs Confidence', 'Confidence', 'F1-Score')

# Confusion matrix
def plot_confusion_matrix(true, pred, labels):
    cm = confusion_matrix(true.argmax(axis=1), pred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(test_labels, predictions, labels=['Pupil', 'Iris'])
