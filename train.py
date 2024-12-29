# train.py
from transformers import (
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config
)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from VAD_dataset import *
import librosa
import matplotlib.pyplot as plt

def calculate_class_weights(dataset, batch_size=32):
    """Calculate class weights from entire dataset using batched processing."""
    print("Calculating class weights...")
    speech_count = 0
    total_count = 0
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Processing dataset"):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_labels = []
        
        for idx in batch_indices:
            labels = dataset[idx]["labels"]
            batch_labels.extend(labels.numpy())
            
        batch_labels = np.array(batch_labels)
        speech_count += np.sum(batch_labels > 0)
        total_count += len(batch_labels)
        
        # Free memory
        del batch_labels
    
    speech_ratio = speech_count / total_count
    nonspeech_ratio = 1 - speech_ratio
    
    print(f"\nDataset statistics:")
    print(f"Speech ratio: {speech_ratio:.4f}")
    print(f"Non-speech ratio: {nonspeech_ratio:.4f}")
    
    # Calculate balanced weights
    weights = torch.tensor([
        1/(nonspeech_ratio + 1e-5),  # weight for non-speech
        1/(speech_ratio + 1e-5)      # weight for speech
    ], dtype=torch.float32)
    
    # Normalize weights to sum to 2
    weights = weights / weights.sum() * 2
    
    return weights

def calculate_metrics(predictions, labels):
    # Convert to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate metrics
    true_positive = np.sum((predictions == 1) & (labels == 1))
    false_positive = np.sum((predictions == 1) & (labels == 0))
    true_negative = np.sum((predictions == 0) & (labels == 0))
    false_negative = np.sum((predictions == 0) & (labels == 1))
    
    # Compute metrics
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'speech_ratio': np.mean(labels == 1)
    }



def train_one_epoch(model, dataloader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        
        # Get model outputs
        outputs = model(input_values=input_values)
        logits = outputs.logits.contiguous()
        
        # Process labels
        labels = labels.float().contiguous()
        n_frames = logits.shape[1]
        frame_size = input_values.shape[1] // n_frames
        labels = labels.unfold(1, frame_size, frame_size)[:, :n_frames].mean(dim=-1) > 0
        labels = labels.long().contiguous()
        
        # Get predictions BEFORE logging statistics
        predictions = torch.argmax(logits, dim=-1)
        
        # Compute weighted loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1),
            weight=class_weights
        )
        print("PRED", predictions.tolist())
        print("TRAIN", labels.tolist())
        
        
        all_preds.extend(predictions.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # metriccs
    metrics = calculate_metrics(all_preds, all_labels)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics



def collate_fn(batch):
    # stack all tensors
    input_values = torch.stack([x["input_values"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    
    return {
        "input_values": input_values,
        "labels": labels,
        "audio_path": [x["audio_path"] for x in batch],
        "num_speakers": [x["num_speakers"] for x in batch],
        "window_idx": [x["window_idx"] for x in batch],
        "start_time": [x["start_time"] for x in batch],
        "end_time": [x["end_time"] for x in batch],
    }



def validate(model, dataloader, device, class_weights):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch_idx, batch in enumerate(progress_bar):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_values=input_values)
            logits = outputs.logits.contiguous()
            
            # Process labels
            labels = labels.float().contiguous()
            n_frames = logits.shape[1]
            frame_size = input_values.shape[1] // n_frames
            labels = labels.unfold(1, frame_size, frame_size)[:, :n_frames].mean(dim=-1) > 0
            labels = labels.long().contiguous()
            
            # compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2),
                labels.reshape(-1),
                weight=class_weights

            )
            
            total_loss += loss.item()
            
            # get predictions
            predictions = torch.argmax(logits, dim=-1)
            print("-" * 50)
            print(predictions)
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Print final distributions
    print("\nFinal Distributions:")
    print(f"Predictions: {np.bincount(np.array(all_preds))}")
    print(f"Labels: {np.bincount(np.array(all_labels))}")
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels)
    avg_loss = total_loss / len(dataloader)
        
    return avg_loss, metrics

def print_metrics(name, metrics):
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")


def check_dataset_stats(dataset):
    all_labels = []
    for i in range(len(dataset)):
        labels = dataset[i]["labels"]
        all_labels.extend(labels.numpy())
    
    labels_array = np.array(all_labels)
    print(f"Total samples: {len(labels_array)}")
    print(f"Unique values: {np.unique(labels_array)}")
    print(f"Value counts: {np.bincount(labels_array.astype(int))}")
    print(f"Speech ratio: {np.mean(labels_array > 0):.4f}")


def main():
    import matplotlib.pyplot as plt
    import os

    # 1. Model setup using Ahmed's model
    config = Wav2Vec2Config.from_pretrained("/nlp/scr/askhan1/wav2vec_ckpt")
    
    # speech and non-speech
    config.num_labels = 2

    # config.hidden_dropout = 0.3       # Add dropout to hidden states
    # config.attention_dropout = 0.3    # Add dropout to attention
    # config.activation_dropout = 0.3   # Add dropout to activation functions
    # config.feat_proj_dropout = 0.3    # Add dropout to feature projection
    # config.layerdrop = 0.3           # Add layer dropout
    
    model = Wav2Vec2ForAudioFrameClassification.from_pretrained(
        "/nlp/scr/askhan1/wav2vec_ckpt", 
        config=config,
        torch_dtype=torch.float32
    )

    # freezing base model and only training classifier
    # model.freeze_base_model()

    # 2. feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, # mono audio
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    # window length for the audio (seconds)
    max_duration = 2
    stride_duration = max_duration / 2
    
    # 3. datasets
    train_dataset = VADDataset(
        manifest_path="./train.json",
        feature_extractor=feature_extractor,
        max_duration_s=max_duration,
        stride_duration_s=stride_duration
    )

    val_dataset = VADDataset(
        manifest_path="./dev.json",
        feature_extractor=feature_extractor,
        max_duration_s=max_duration,
        stride_duration_s=stride_duration
    )


    # 4. Since theres an imbalance of data for speech and non-speech, pre-calculated weights on the entire training dataset
    class_weights = torch.tensor([1.5, 0.5])
    print("Class weights:", class_weights)

    # 5. dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  
        shuffle=True,
        num_workers=4,  
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,  
        shuffle=False,
        num_workers=4,  
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 6. train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_weights = class_weights.to(device)
    
    num_epochs = 100

    optimizer = AdamW(model.parameters(), lr=0.00001)
   
    # Initialize metrics storage
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }

    # Create output directory for plots
    os.makedirs("training_plots", exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        print(f"Training - Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}")

        # Update training metrics
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_metrics['f1'])
        
        # Validation
        if epoch % 10 == 0:
            val_loss, raw_metrics = validate(model, val_loader, device, class_weights)
            val_f1 = raw_metrics['f1']
            print(f"Validation - Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
            
            # Update validation metrics
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
        else:
           
            if len(history['val_loss']) > 0:
                history['val_loss'].append(history['val_loss'][-1])
                history['val_f1'].append(history['val_f1'][-1])

            #    plot metrics
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, epoch + 2), history['train_loss'], label='Train Loss', marker='o')
                plt.plot(range(1, epoch + 2), history['val_loss'], label='Validation Loss', marker='x')
                plt.plot(range(1, epoch + 2), history['train_f1'], label='Train F1', marker='o')
                plt.plot(range(1, epoch + 2), history['val_f1'], label='Validation F1', marker='x')
                plt.xlabel('Epoch')
                plt.ylabel('Metrics')
                plt.title('Training and Validation Metrics')
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.savefig(f"training_plots/epoch.png")
                plt.close()
        

    

if __name__ == "__main__":
   main()
