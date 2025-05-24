import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import os
import cv2
import numpy as np
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
IMAGE_DIR = './images'
CSV_PATH = './nodes_labeled_easyocr.csv'  # Assuming this is the CSV file with labels
IMG_SIZE = 28
NUM_CLASSES = 6 # 0 to 5
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
N_FOLDS = 5

# Custom Dataset
class DigitDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to required dimensions
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalize pixel values to [0,1]
        img = img / 255.0

        # Convert to torch tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return img_tensor, label

# Define our CNN with Min-Pooling (inverse of MaxPooling for white background)
class MinPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MinPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        # Invert the image (1 - x) to make min values become max
        inverted = 1 - x
        # Use max pooling on inverted image
        pooled = nn.functional.max_pool2d(inverted, self.kernel_size,
                                          self.stride, self.padding)
        # Invert back
        return 1 - pooled

class DigitCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DigitCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.min_pool1 = MinPooling(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.min_pool2 = MinPooling(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.min_pool3 = MinPooling(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.min_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.min_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.min_pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def load_data():
    """Load images and labels from CSV."""
    df = pd.read_csv(CSV_PATH)

    # Assuming CSV has columns 'filename' and 'label'
    image_paths = [os.path.join(IMAGE_DIR, filename) for filename in df['filename']]
    labels = df['key'].values

    # Print class distribution
    label_counts = Counter(labels)
    print("Class distribution:", label_counts)

    return image_paths, labels

def compute_class_weights(labels):
    """Compute weights for each class for dealing with class imbalance."""
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    class_weights = total / (NUM_CLASSES * class_counts)
    return torch.FloatTensor(class_weights)

def train_and_validate():
    """Train the model using stratified K-fold cross-validation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    image_paths, labels = load_data()

    # Compute class weights for handling imbalance
    class_weights = compute_class_weights(labels)
    print("Class weights:", class_weights)

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_accuracies = []
    best_model = None
    best_accuracy = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        print(f"\nTraining fold {fold+1}/{N_FOLDS}")

        # Split data for this fold
        train_paths = [image_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Create datasets
        train_dataset = DigitDataset(train_paths, train_labels)
        val_dataset = DigitDataset(val_paths, val_labels)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Initialize model, loss function, and optimizer
        model = DigitCNN().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss = train_loss / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            scheduler.step(val_loss)

            val_accuracy = accuracy_score(val_targets, val_preds)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Accuracy: {val_accuracy:.4f}")

        # Final validation metrics for this fold
        val_accuracy = accuracy_score(val_targets, val_preds)
        print(f"Fold {fold+1} Validation Accuracy: {val_accuracy:.4f}")
        print(classification_report(val_targets, val_preds))

        fold_accuracies.append(val_accuracy)

        # Save best model across folds
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model

    print(f"\nCross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Deviation: {np.std(fold_accuracies):.4f}")

    # Save the best model
    torch.save(best_model.state_dict(), 'best_digit_classifier.pth')
    print(f"Best model saved with accuracy: {best_accuracy:.4f}")

    return best_model

def predict_from_numpy(img_array, model=None):
    """Predict digit from a numpy array."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model if not provided
    if model is None:
        model = DigitCNN()
        model.load_state_dict(torch.load('best_digit_classifier.pth'))
        model.to(device)

    model.eval()

    # Preprocess image array
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    # Convert to grayscale if it's RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img / 255.0
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(prob, 1)

    return predicted.item(), confidence.item()

if __name__ == "__main__":
    # Train the model
    best_model = train_and_validate()

    # Example of how to use the prediction function with a numpy array
    print("\nExample of prediction from numpy array:")

    sample_image = cv2.imread("images/node_00016.png")
    sample_image = sample_image.astype(np.uint8)

    digit, confidence = predict_from_numpy(sample_image)
    print(f"Predicted digit: {digit}, Confidence: {confidence:.4f}")