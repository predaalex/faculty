#%%
import builtins
from datetime import datetime


def print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{timestamp}] ", *args, **kwargs)
#%%
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
#%%
ds = load_dataset("microsoft/cats_vs_dogs")
#%%
from datasets import DatasetDict

train_test_split = ds["train"].train_test_split(test_size=0.2, seed=42)
test_valid_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

# Create new dataset dictionary
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": test_valid_split["train"],
    "test": test_valid_split["test"]
})
#%%
class cats_vs_dogs(Dataset):
    def __init__(self, dataset, transform, stage="train"):
        self.dataset = dataset
        self.transform = transform
        self.stage = stage
    def __len__(self):
        return len(self.dataset[self.stage])
        
    def __getitem__(self, idx):
        image = self.dataset[self.stage][idx]['image'].convert("RGB")
        label = self.dataset[self.stage][idx]['labels']
        
        label = torch.tensor(label).float()
        image = self.transform(image)
        
        return image, label
#%%
transform_config = {
    'train': transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
}
#%%
train_dataset = cats_vs_dogs(dataset, transform_config["train"], stage="train")
test_dataset = cats_vs_dogs(dataset, transform_config["test"], stage="test")
validation_dataset = cats_vs_dogs(dataset, transform_config["test"], stage="validation")
#%%
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
#%%
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.squeeze(1)
        return x
#%%
class VGG(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            *self.cnn_block(3, 64, 2),
            *self.cnn_block(64, 128, 2),
            *self.cnn_block(128, 256, 3),
            *self.cnn_block(256, 512, 3),
            *self.cnn_block(512, 512, 3),
        )              
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        
    def cnn_block(self, in_channels, out_channels, nr_layers):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(nr_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return layers
    
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.squeeze(1)
        return x
#%%
import torch
import torch.nn as nn

# 1. Patch Embedding: Split image into non-overlapping patches and project them to an embedding space.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        img_size: size of the input image (assumed square).
        patch_size: size of each patch.
        in_chans: number of input channels (3 for RGB).
        embed_dim: dimension of the embedding space for each patch.
        """
        super().__init__()
        self.patch_size = patch_size
        # This convolution uses kernel size and stride equal to patch_size.
        # It converts the image into a grid of patches and projects each patch into an embedding vector.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Calculate the number of patches (e.g., for 224x224 with 16x16 patches, we have 14x14 = 196 patches)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x has shape (batch_size, 3, 224, 224)
        x = self.proj(x)  # Now shape becomes (batch_size, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # Flatten the height and width dimensions; shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Transpose to shape: (batch_size, num_patches, embed_dim)
        return x

# 2. MLP Block: A simple feed-forward network used inside the mixer layers.
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        """
        in_features: size of each input vector.
        hidden_features: hidden layer dimension.
        out_features: size of the output vector.
        dropout: dropout rate applied after each linear layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)  # First linear layer
        self.act = nn.GELU()  # Activation function
        self.fc2 = nn.Linear(hidden_features, out_features)  # Second linear layer
        self.drop = nn.Dropout(dropout)  # Dropout layer for regularization
        
    def forward(self, x):
        x = self.fc1(x)   # Apply first linear layer
        x = self.act(x)   # Apply GELU activation
        x = self.drop(x)  # Apply dropout
        x = self.fc2(x)   # Apply second linear layer
        x = self.drop(x)  # Apply dropout again
        return x

# 3. Mixer Layer: Contains both token mixing and channel mixing MLPs with residual connections.
class MixerLayer(nn.Module):
    def __init__(self, num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout=0.0):
        """
        num_patches: number of patches (tokens) per image.
        embed_dim: dimension of each patch embedding.
        token_mlp_dim: hidden dimension for the token-mixing MLP.
        channel_mlp_dim: hidden dimension for the channel-mixing MLP.
        dropout: dropout rate for the MLPs.
        """
        super().__init__()
        # Normalization before token mixing
        self.norm1 = nn.LayerNorm(embed_dim)
        # Token mixing MLP works across the tokens dimension. It first transposes the input.
        self.token_mixing = Mlp(num_patches, token_mlp_dim, num_patches, dropout)
        
        # Normalization before channel mixing
        self.norm2 = nn.LayerNorm(embed_dim)
        # Channel mixing MLP works on each token independently along the channel dimension.
        self.channel_mixing = Mlp(embed_dim, channel_mlp_dim, embed_dim, dropout)
        
    def forward(self, x):
        # x shape: (batch_size, num_patches, embed_dim)
        
        # ---- Token Mixing ----
        y = self.norm1(x)  # Normalize across the embedding dimension
        # Transpose to swap tokens and channels: shape becomes (batch_size, embed_dim, num_patches)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)  # Mix information across the tokens (spatial locations)
        # Transpose back to original shape: (batch_size, num_patches, embed_dim)
        y = y.transpose(1, 2)
        # Add residual connection
        x = x + y
        
        # ---- Channel Mixing ----
        y = self.norm2(x)  # Normalize before channel mixing
        y = self.channel_mixing(y)  # Mix information along the channels for each token
        # Add residual connection
        x = x + y
        return x

# 4. MLP-Mixer Model: Stack the patch embedding, a series of Mixer layers, and the classification head.
class MLPMixer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dim=768, depth=12, dropout=0.0):
        """
        img_size: size of the input image.
        patch_size: size of each patch.
        in_chans: number of image channels.
        num_classes: number of output classes (2 for cats vs. dogs).
        embed_dim: dimension of patch embeddings.
        depth: number of Mixer layers to stack.
        dropout: dropout rate for the MLP blocks.
        """
        super().__init__()
        # Step 1: Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Compute the number of patches (tokens)
        num_patches = self.patch_embed.num_patches  # For 224x224 and 16x16 patches, num_patches = 196
        
        # For the token-mixing MLP, the hidden dimension is set to the number of patches.
        token_mlp_dim = num_patches
        # For the channel-mixing MLP, the hidden dimension is 4 times the embedding dimension.
        channel_mlp_dim = embed_dim * 4
        
        # Step 2: Stack Mixer layers
        self.mixer_layers = nn.Sequential(*[
            MixerLayer(num_patches, embed_dim, token_mlp_dim, channel_mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Step 3: Final normalization layer
        self.norm = nn.LayerNorm(embed_dim)
        
        # Step 4: Classification head.
        # After processing through Mixer layers, we perform global average pooling over tokens
        # and then project the resulting vector to the number of classes.
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224)
        
        # Convert image to patch embeddings (tokens)
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)
        
        # Process the tokens through the Mixer layers
        x = self.mixer_layers(x)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        # Global average pooling over the token dimension
        x = x.mean(dim=1)  # Now shape: (batch_size, embed_dim)
        
        # Final classifier to obtain logits for each class (cats vs. dogs)
        x = self.classifier(x)  # Shape: (batch_size, num_classes)
        return x.squeeze(1)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
model = AlexNet(num_classes=1)
model.load_state_dict(torch.load("models/best_models/AlexNet_Best+lr=0.005+eta_min=0.0005+weight_decay=5e-05.pt"))
# Dummy input tensor of size (batch_size, channels, height, width)
input_tensor = torch.randn(2, 3, 224, 224)
output = model(input_tensor)
print(output)
#%%
model = VGG(num_classes=1, dropout=0.2)
# Dummy input tensor of size (batch_size, channels, height, width)
input_tensor = torch.randn(2, 3, 224, 224)
output = model(input_tensor)
print(output)
#%%
model = MLPMixer(img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dim=768, depth=12, dropout=0.1)
input_tensor = torch.randn(2, 3, 224, 224)
output = model(input_tensor)
print(output)
#%%
def initialize_weights(m):
    """
    Initializes the weights of the MLP-Mixer model.
    This function is applied recursively to all layers in the model.
    """
    if isinstance(m, nn.Linear):
        # Xavier Normal Initialization for Linear layers (MLPs)
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Initialize bias to zero for stability
    
    elif isinstance(m, nn.LayerNorm):
        # Initialize LayerNorm (scale=1, bias=0)
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Conv2d):
        # Kaiming Normal Initialization for the Patch Embedding layer (Conv2d)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)
model.to(device)
#%%
from torch.nn import BCEWithLogitsLoss

criterion = BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,
    T_mult=1,
    eta_min=5e-4,
)
model_config = {
    "lr": 1e-2,
    "eta_min": 5e-4,
    "weight_decay": 5e-4,
}
#%%
from sklearn.metrics import confusion_matrix as cf_mx
import seaborn as sns

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def validation_method(criterion, model, val_loader, confusion_matrix=False):
    ### VALIDATING
    model.eval()
    validation_loss = 0.0
    all_labels = []  # Ground truth labels for validation
    all_preds = []  # Predictions for validation
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            output = model(images)  # Forward pass (logits)
            loss = criterion(output, labels)  # Compute validation loss
            validation_loss += loss.item()

            # Convert logits to probabilities and apply threshold
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            # Store for statistics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    val_loss = validation_loss / len(val_loader)  # Average validation loss
    
    # Compute validation statistics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Compute confusion matrix if requested
    if confusion_matrix:
        conf_matrix = cf_mx(all_labels, all_preds)
        plot_confusion_matrix(conf_matrix)
    
    return val_accuracy, val_f1, val_loss, val_precision, val_recall
#%%
from matplotlib import pyplot as plt
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


def training_method(model, criterion, optimizer, scheduler, train_loader, val_loader, model_config, num_epochs=50, patience=5, delta = 0.02, loss_procentage_improvement=10, model_name="vgg"):
    train_losses = []  # List to store training losses
    val_losses = []  # List to store validation losses
    val_accuracies = []  # List to store validation accuracies
    val_precisions = []  # List to store validation precisions
    val_recalls = []  # List to store validation recalls
    val_f1s = []  # List to store validation F1-scores
    learning_rates = [] # List to store learning rate progression

    best_val_loss = float('inf')  # Initialize the best validation loss
    initial_loss = float('inf')
    best_model = None  # Store the best model
    epochs_without_improvement = 0  # Track epochs without improvement

    for epoch in range(num_epochs):
        start_time = time.time()
        ### TRAINING
        model.train()
        training_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            output = model(images)  # Forward pass (logits)
            loss = criterion(output, labels)  # Compute loss
            loss.backward()  # Backpropagation
            
            optimizer.step()  # Update weights
            training_loss += loss.item()  # Accumulate loss
            
            if batch_idx % (len(train_loader) // 4) == 0 and batch_idx != 0:
                print(f"[{epoch}, {batch_idx}/{len(train_loader)}] Loss: {training_loss / batch_idx:.4f}")
        
        train_loss = training_loss / len(train_loader)  # Average training loss
        train_losses.append(train_loss)

        val_accuracy, val_f1, val_loss, val_precision, val_recall = validation_method(criterion, model, val_loader)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        
        if epoch == 1:
            initial_loss = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)  # Save the best model
            epochs_without_improvement = 0  # Reset counter
            print(f"New best model with Loss: {val_loss:.4f} at epoch {epoch + 1}")
        elif val_loss < best_val_loss + delta:
            print(f"Validation loss did not improve significantly")            
        else:
            epochs_without_improvement += 1
            print(f"Validation loss did not improve for {epochs_without_improvement} epoch(s).")
            # Stop training if validation loss does not improve for 'patience' epochs
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best Loss: {best_val_loss:.4f}")
                break  # Exit training loop


        # Step the learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']  # Get the current learning rate
        learning_rates.append(current_lr)
        end_time = time.time()

        print(f"\nEpoch {epoch + 1}/{num_epochs} - "
              f"Training Loss: {train_loss:.4f} - "
              f"Validation Loss: {val_loss:.4f} - "
              f"Accuracy: {val_accuracy:.4f} - "
              f"Precision: {val_precision:.4f} - "
              f"Recall: {val_recall:.4f} - "
              f"F1 Score: {val_f1:.4f} - "
              f"Time: {end_time - start_time:.2f} - "
              f"Lr: {current_lr:.2e}")

    print('Training finished!')
    
    # save the model only if the best loss is lower than the first initial loss ( to see that the model actually improved with 10% loss )
    if best_val_loss < (100 - loss_procentage_improvement) * initial_loss:
        # Init plot&model save path
        plt_save_path = f"models/{model_name}"
        for key, value in model_config.items():
            plt_save_path += key + "=" + str(value) + "+"
        plt_save_path = plt_save_path[:-1] + ".png"
        model_path = plt_save_path[:-4] + ".pt"
        torch.save(best_model.state_dict(), model_path)
        print(f"Best model with Loss: {best_val_loss:.4f} saved.")
        print(f"Model saved to {model_path}")

        # Plotting the losses and validation metrics over epochs
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(val_accuracies, label='Accuracy')
        plt.plot(val_precisions, label='Precision')
        plt.plot(val_recalls, label='Recall')
        plt.plot(val_f1s, label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('Metric')
        plt.title('Validation Metrics')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(learning_rates, label='Learning Rate')
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Progression")
        plt.legend()
    
        plt.tight_layout()
        plt.savefig(plt_save_path)
        plt.show()
    
    else:
        print(f"Model wasn't saved because it didn't improve: {loss_procentage_improvement}%")
    
    return best_model
#%%
best_model = training_method(model, criterion, optimizer, scheduler, train_dataloader, validation_dataloader, model_config, num_epochs=1, model_name="MLPMixer")
#%%
val_accuracy, val_f1, val_loss, val_precision, val_recall = validation_method(criterion, best_model, validation_dataloader, confusion_matrix=True)
#%% md
# # Hyperparameter tuning
#%%
import gc


def hyperparameter_tuning(model_class, parameters_grid, epochs, train_dataloader, validation_dataloader, results, model_name="AlexNet"):
    best_model = None
    best_val_loss = np.inf
    for idx in tqdm(range(len(parameters_grid["lr"]))):
        # get params
        lr = parameters_grid["lr"][idx]
        eta_min = parameters_grid["eta_min"][idx]
        weight_decay = random.choice(parameters_grid['weight_decay'])
        model_config = {
            "lr": lr,
            "eta_min": eta_min,
            "weight_decay": weight_decay,
        }
        print(f"Learning rate {lr:.2e} - "
              f"eta_min {eta_min:.2e} - "
              f"weight_decay {weight_decay:.2e}")
        try:
            model = model_class(num_classes=1)
            if model_name =="vgg":
                model.apply(initialize_weights)
            model.to(device)
            
            criterion = BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,
                T_mult=1,
                eta_min=eta_min,
            )
            
            best_curr_model = training_method(model, criterion, optimizer, scheduler, train_dataloader, validation_dataloader, model_config, num_epochs=epochs, model_name=model_name)
            
            val_accuracy, val_f1, val_loss, val_precision, val_recall = validation_method(criterion, best_curr_model, validation_dataloader, confusion_matrix=True)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(best_curr_model)
            
            result = {
                "lr": lr,
                "eta_min": eta_min,
                "weight_decay": weight_decay,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }
            results.append(result)
            
        except RuntimeError as e:
            print(f"Error with configuration: {lr=}, {eta_min=}, {weight_decay=}")
            print(f"Error message: {str(e)}")
        
        finally:
            # Reset GPU memory
            print("Resetting GPU memory...")
            torch.cuda.empty_cache()
            gc.collect()
    
    return best_model, results
#%% md
# # AlexNet
#%%
param_grid = {
    "lr":       [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    "eta_min":  [1e-4, 1e-4, 5e-5, 5e-5, 1e-5],
    "weight_decay": [5e-4, 1e-5, 5e-5]
}


total_combination = len(param_grid['lr'])
epochs = 30
time_per_epoch = 140 # approx

print(f"Total time to hyper tune: {total_combination * epochs * time_per_epoch / 3600:.2f} hours")
#%%
AlexNet_results = []
#%%
best_AlexNet_model, AlexNet_results = hyperparameter_tuning(AlexNet, param_grid, epochs, train_dataloader, validation_dataloader, AlexNet_results, model_name="AlexNet")
#%%
import pandas as pd

results_df = pd.DataFrame(AlexNet_results)
results_df.fillna("", inplace=True)
results_df.sort_values(by="val_loss", inplace=True)
results_df
#%% md
# # VGG
#%%
param_grid = {
    "lr":       [1e-2, 1e-3, 1e-4],
    "eta_min":  [1e-3, 1e-4, 1e-5],
    "weight_decay": [1e-5]
}


total_combination = len(param_grid['lr'])
epochs = 15
time_per_epoch = 240 # approx

print(f"Total time to hyper tune: {total_combination * epochs * time_per_epoch / 3600:.2f} hours")
#%%
VGG_results = []
#%%
best_VGG_model, VGG_results = hyperparameter_tuning(VGG, param_grid, epochs, train_dataloader, validation_dataloader, VGG_results, model_name="vgg")
#%%
import pandas as pd

results_df = pd.DataFrame(VGG_results)
results_df.fillna("", inplace=True)
results_df.sort_values(by="val_loss", inplace=True)
results_df
#%% md
# # MLP Mixer
#%%
param_grid = {
    "lr":       [1e-2, 1e-3, 1e-4],
    "eta_min":  [1e-3, 1e-4, 1e-5],
    "weight_decay": [1e-4, 1e-5]
}


total_combination = len(param_grid['lr'])
epochs = 15
time_per_epoch = 230 # approx

print(f"Total time to hyper tune: {total_combination * epochs * time_per_epoch / 3600:.2f} hours")
#%%
MLP_Mixer_results = []
#%%
best_MLP_Mixer_model, MLP_Mixer_results = hyperparameter_tuning(VGG, param_grid, epochs, train_dataloader, validation_dataloader, MLP_Mixer_results, model_name="vgg")
#%%
import pandas as pd

results_df = pd.DataFrame(VGG_results)
results_df.fillna("", inplace=True)
results_df.sort_values(by="val_loss", inplace=True)
results_df
#%% md
# # View wrong classified images
#%%
import torch
import os
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


criterion = nn.BCEWithLogitsLoss()

# Ensure evaluation mode
tested_model = AlexNet(num_classes=1)
tested_model.load_state_dict(torch.load("models/best_models/AlexNet_Best+lr=0.005+eta_min=0.0005+weight_decay=5e-05.pt"))
tested_model.to(device)
tested_model.eval()
validation_loss = 0.0
all_labels = []  # Ground truth labels for validation
all_preds = []  # Predictions for validation

# Directory to save misclassified images
misclassified_dir = "misclassified_images_VGG"
os.makedirs(misclassified_dir, exist_ok=True)

misclassified_samples = []  # Store tuples of (image path, predicted label, true label)

# Define the inverse transform
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(validation_dataloader):
        images, labels = images.to(device), labels.to(device)

        output = tested_model(images)  # Forward pass (logits)
        loss = criterion(output, labels)  # Compute validation loss
        validation_loss += loss.item()

        # Convert logits to probabilities and apply threshold
        probs = torch.sigmoid(output)
        preds = (probs > 0.5).float()

        # Store for statistics
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        # Identify misclassified images
        incorrect_indices = (preds != labels).cpu().numpy().astype(bool)
        incorrect_images = images[incorrect_indices]
        incorrect_preds = preds[incorrect_indices].cpu().numpy()
        incorrect_labels = labels[incorrect_indices].cpu().numpy()

        # Reverse normalization
        incorrect_images = incorrect_images.cpu() * std + mean  # Unnormalize

        # Save misclassified images
        for i, (img, pred_label, true_label) in enumerate(zip(incorrect_images, incorrect_preds, incorrect_labels)):
            img_pil = to_pil_image(img.clamp(0, 1))  # Convert tensor to PIL image
            filename = f"{misclassified_dir}/batch{batch_idx}_img{i}_pred{int(pred_label)}_true{int(true_label)}.png"
            img_pil.save(filename)
            misclassified_samples.append((filename, pred_label, true_label))

# Print misclassified images with their labels
print("Misclassified Images:")
for filename, pred_label, true_label in misclassified_samples:
    print(f"{filename} -> Predicted: {pred_label}, True: {true_label}")
#%%
