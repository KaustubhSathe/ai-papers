import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import argparse
import pickle  # For saving/loading vocabulary
import time   # Import the time module

# Import model definition
from model import EncoderCNN, DecoderRNN

# Import data loading components (REAL IMPLEMENTATION)
from build_vocab import Vocabulary # To load the vocab class structure
from data_loader import get_loader # Import the function to create the data loader

# --- Configuration ---
class Config:
    # Paths
    model_path = 'models/'
    log_path = 'logs/'
    # Assume COCO data is structured like: data_path/train2017 and data_path/annotations/captions_train2017.json
    # Adjust these paths based on where you downloaded the COCO dataset relative to your project root
    data_path = './' # Base path containing annotations dir
    image_dir_relative_to_data_path = './train2017/train2017' # Relative path to images from data_path (adjust if needed)
    caption_file = 'annotations/captions_train2017.json' # Relative path to annotation json within data_path
    vocab_path = './vocab.pkl' # Path to load vocabulary

    # Model Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1

    # Training Hyperparameters
    num_epochs = 5
    batch_size = 128 # Adjust based on GPU memory
    learning_rate = 0.001
    log_step = 100       # Log training status every N steps
    save_step = 1000     # Save model checkpoint every N steps

    # Data Loader parameters
    num_workers = 2 # Adjust based on your system

def main(config):
    # Create necessary directories
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Image preprocessing/transformations
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ])

    # Load Vocabulary wrapper
    try:
        # Ensure Vocabulary class definition is available
        # This happens via 'from build_vocab import Vocabulary'
        with open(config.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {config.vocab_path}")
        vocab_size = len(vocab)
    except FileNotFoundError:
         print(f"Error: Vocabulary file not found at {config.vocab_path}.")
         print("Please run build_vocab.py first.")
         return # Exit if vocab not found
    except Exception as e:
         print(f"Error loading vocabulary: {e}")
         import traceback
         traceback.print_exc()
         return

    # Build data loader using the actual get_loader function
    print("Building data loader...")
    # Construct full paths from config
    # Note: Adjust path joining logic if your structure differs significantly
    caption_json_path = os.path.join(config.data_path, config.caption_file)
    # Construct image root path based on the relationship defined in config
    image_root = os.path.abspath(os.path.join(config.data_path, config.image_dir_relative_to_data_path))

    print(f"Attempting to load images from: {image_root}")
    print(f"Attempting to load captions from: {caption_json_path}")

    if not os.path.exists(image_root):
        print(f"Error: Image directory not found at {image_root}")
        return
    if not os.path.exists(caption_json_path):
        print(f"Error: Caption file not found at {caption_json_path}")
        return

    data_loader = get_loader(
        root=image_root,
        json_path=caption_json_path,
        vocab=vocab,
        transform=transform,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    print("Data loader created.")
    # --- End Data Loader (Removed Dummy Loader) ---


    # --- Model Initialization ---
    encoder = EncoderCNN(config.embed_size).to(device)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, vocab_size, config.num_layers).to(device)
    print("Models initialized.")

    # --- Loss and Optimizer ---
    # Use ignore_index for padding token in CrossEntropyLoss
    pad_idx = vocab(vocab.word2idx['<pad>']) # Get the index of the pad token
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # Combine parameters from both decoder and the trainable parts of encoder
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=config.learning_rate)
    print("Loss and optimizer set up.")

    # --- Training Loop ---
    print("Starting Training...")
    start_time = time.time() # Record start time
    total_steps = len(data_loader)
    for epoch in range(config.num_epochs):
        for i, batch_data in enumerate(data_loader):
            # Handle potential None batches from collate_fn if filtering occurred
            if batch_data is None or batch_data[0] is None:
                print(f"Skipping empty/invalid batch at step {i+1}")
                continue

            images, captions, lengths = batch_data

            # Move data to the configured device
            images = images.to(device)
            captions = captions.to(device)
            # lengths tensor is usually kept on CPU for pack_padded_sequence
            lengths = lengths.cpu() # Ensure lengths are on CPU

            # Prepare targets and decoder inputs
            # Targets: sequence excluding <start> token
            targets = captions[:, 1:]
            # Decoder inputs: sequence excluding <end> token
            decoder_inputs = captions[:, :-1]
            # Lengths for packing should correspond to decoder_inputs lengths (-1 from original)
            input_lengths = lengths - 1

            # Filter out sequences that become too short (length 0) after removing token
            valid_indices = input_lengths > 0
            if not valid_indices.all():
                # Filter batch elements based on valid lengths
                images = images[valid_indices]
                captions = captions[valid_indices] # Keep original captions for reference if needed
                targets = targets[valid_indices]
                decoder_inputs = decoder_inputs[valid_indices]
                input_lengths = input_lengths[valid_indices]
                lengths = lengths[valid_indices] # Keep original lengths consistent
                if images.nelement() == 0: # Check if batch became empty
                    print(f"Batch became empty after filtering short captions at step {i+1}. Skipping.")
                    continue
                print(f"Filtered {sum(~valid_indices)} short captions at step {i+1}")


            # Pack the target sequences (excluding <start>) for loss calculation
            # Ensure lengths tensor is on CPU
            targets_packed = nn.utils.rnn.pack_padded_sequence(targets, input_lengths, batch_first=True, enforce_sorted=False)[0]

            # Zero the gradients
            decoder.zero_grad()
            encoder.zero_grad() # Zero gradients for encoder's trainable parts

            # Forward pass
            features = encoder(images)
            # Pass decoder_inputs (excluding <end>) and their lengths to the decoder
            outputs = decoder(features, decoder_inputs, input_lengths) # Lengths should be on CPU

            # Calculate the loss using packed outputs and packed targets
            loss = criterion(outputs, targets_packed)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # --- Added Log at every step ---
            step = i + 1 # Use step consistently (starts from 1)
            # Basic log at every single step
            # print(f'E[{epoch+1}/{config.num_epochs}] S[{step}/{total_steps}] Loss: {loss.item():.4f}') # Optional: Comment out if too verbose
            # --- End Added Log ---


            # Log training status periodically (keep existing log for less frequent, more detailed info)
            if step % config.log_step == 0:
                perplexity = torch.exp(loss).item() if loss.item() < 100 else float('inf') # Avoid overflow
                print(f'--- Log Step --- Epoch [{epoch+1}/{config.num_epochs}], Step [{step}/{total_steps}], '
                      f'Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}')

            # Save model checkpoints
            if step % config.save_step == 0:
                encoder_path = os.path.join(config.model_path, f'encoder-{epoch+1}-{step}.ckpt')
                decoder_path = os.path.join(config.model_path, f'decoder-{epoch+1}-{step}.ckpt')
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                print(f"--- Checkpoint Saved --- Models to {encoder_path} and {decoder_path}")

        # Optional: Add end-of-epoch saving or validation here
        print(f"--- End of Epoch {epoch+1} ---")


    print("Training finished.")
    end_time = time.time() # Record end time
    total_training_time = end_time - start_time # Calculate duration

    # --- Print Total Training Time ---
    print(f"Total training time: {total_training_time:.2f} seconds")
    # Convert to minutes and hours for readability
    minutes, seconds = divmod(total_training_time, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    # --- End Print Total Training Time ---

    # Final save
    encoder_path = os.path.join(config.model_path, 'encoder-final.ckpt')
    decoder_path = os.path.join(config.model_path, 'decoder-final.ckpt')
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Saved final models to {encoder_path} and {decoder_path}")


if __name__ == '__main__':
    # You might want to use argparse for more flexible configuration
    # parser = argparse.ArgumentParser()
    # Add arguments for paths, hyperparameters etc.
    # args = parser.parse_args()
    # config = Config() # Update config based on args
    config = Config()
    main(config) 