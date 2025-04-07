import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import argparse
import pickle  # For saving/loading vocabulary
import time   # Import the time module
import glob   # For finding checkpoint files
import re     # For parsing checkpoint filenames

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

    # --- Add Checkpoint Resume ---
    resume = None # Path prefix to checkpoint file or 'latest'
    # --- End Checkpoint Resume ---

def find_latest_checkpoint(model_path):
    """Finds the latest checkpoint files based on epoch and step."""
    encoder_pattern = os.path.join(model_path, 'encoder-*-*.ckpt')
    decoder_pattern = os.path.join(model_path, 'decoder-*-*.ckpt')
    optimizer_pattern = os.path.join(model_path, 'optimizer-*-*.ckpt') # Add optimizer pattern

    encoder_files = glob.glob(encoder_pattern)
    decoder_files = glob.glob(decoder_pattern)
    optimizer_files = glob.glob(optimizer_pattern) # Find optimizer files

    if not encoder_files or not decoder_files or not optimizer_files:
        return None, None, None, 0, 0 # No checkpoints found

    latest_epoch = -1
    latest_step = -1
    latest_encoder_file = None
    latest_decoder_file = None
    latest_optimizer_file = None

    # Regex to extract epoch and step
    pattern = re.compile(r'.*-(\d+)-(\d+)\.ckpt')

    # Check encoder files (assuming naming convention is consistent)
    for f in encoder_files:
        match = pattern.match(f)
        if match:
            epoch = int(match.group(1))
            step = int(match.group(2))

            # Check if corresponding decoder and optimizer exist
            decoder_f = f.replace('encoder-', 'decoder-')
            optimizer_f = f.replace('encoder-', 'optimizer-')

            if decoder_f in decoder_files and optimizer_f in optimizer_files:
                 if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_step = step
                    latest_encoder_file = f
                    latest_decoder_file = decoder_f
                    latest_optimizer_file = optimizer_f
                 elif epoch == latest_epoch and step > latest_step:
                    latest_step = step
                    latest_encoder_file = f
                    latest_decoder_file = decoder_f
                    latest_optimizer_file = optimizer_f


    if latest_encoder_file:
        print(f"Found latest checkpoint: Epoch {latest_epoch}, Step {latest_step}")
        # Adjust epoch back to 0-based index for loop range
        # Return 1-based epoch and step as stored in filename
        return latest_encoder_file, latest_decoder_file, latest_optimizer_file, latest_epoch, latest_step
    else:
        print("No complete checkpoint set (encoder, decoder, optimizer) found matching the pattern.")
        return None, None, None, 0, 0


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

    # --- Checkpoint Resuming Logic ---
    start_epoch = 0
    start_step = 0 # Step index within the epoch (0-based)
    checkpoint_loaded = False

    if config.resume:
        if config.resume == 'latest':
            print("Attempting to resume from the latest checkpoint...")
            encoder_path, decoder_path, optimizer_path, loaded_epoch, loaded_step = find_latest_checkpoint(config.model_path)
        else:
            # --- Corrected logic for specific path ---
            print(f"Attempting to resume from specified checkpoint path: {config.resume}")
            # Assume the provided path is one of the components (e.g., encoder)
            # Derive the others by replacing the prefix in the original path.
            if 'encoder-' in config.resume:
                encoder_path = config.resume
                decoder_path = config.resume.replace('encoder-', 'decoder-', 1)
                optimizer_path = config.resume.replace('encoder-', 'optimizer-', 1)
            elif 'decoder-' in config.resume:
                decoder_path = config.resume
                encoder_path = config.resume.replace('decoder-', 'encoder-', 1)
                optimizer_path = config.resume.replace('decoder-', 'optimizer-', 1)
            elif 'optimizer-' in config.resume:
                optimizer_path = config.resume
                encoder_path = config.resume.replace('optimizer-', 'encoder-', 1)
                decoder_path = config.resume.replace('optimizer-', 'decoder-', 1)
            else:
                print(f"Error: Could not determine checkpoint type from path: {config.resume}")
                encoder_path, decoder_path, optimizer_path = None, None, None

            # Extract epoch and step from the filename part of the provided path
            filename = os.path.basename(config.resume) # e.g., encoder-3-1000.ckpt
            match = re.search(r'-(\d+)-(\d+)\.ckpt$', filename)
            if match:
                loaded_epoch = int(match.group(1))
                loaded_step = int(match.group(2))
                print(f"Parsed Epoch: {loaded_epoch}, Step: {loaded_step} from path.")
            else:
                print(f"Warning: Could not parse epoch/step from filename: {filename}. Resuming from epoch 0, step 0.")
                loaded_epoch = 0
                loaded_step = 0
            # --- End Corrected logic ---


        if encoder_path and decoder_path and optimizer_path and os.path.exists(encoder_path) and os.path.exists(decoder_path) and os.path.exists(optimizer_path):
            print(f"Loading encoder from: {encoder_path}")
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            print(f"Loading decoder from: {decoder_path}")
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            print(f"Loading optimizer from: {optimizer_path}")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

            # loaded_epoch is 1-based from filename, adjust for 0-based loop
            start_epoch = loaded_epoch -1
            # loaded_step is 1-based from filename, step loop starts from 1, but enumerate is 0-based
            # We want to start *after* the loaded step
            start_step = loaded_step # The next step to run will be start_step + 1

            print(f"Resuming training from Epoch {start_epoch + 1}, Step {start_step + 1}")
            checkpoint_loaded = True
        else:
            print(f"Checkpoint files not found for resume='{config.resume}'. Starting training from scratch.")
    else:
        print("No resume flag provided. Starting training from scratch.")
    # --- End Checkpoint Resuming Logic ---


    # --- Training Loop ---
    print("Starting Training...")
    start_time = time.time() # Record start time
    total_steps = len(data_loader)
    # Adjust epoch range based on loaded checkpoint
    for epoch in range(start_epoch, config.num_epochs):
        # If resuming mid-epoch, skip steps already completed
        epoch_start_step = start_step if epoch == start_epoch and checkpoint_loaded else 0
        if epoch_start_step > 0:
             print(f"Resuming Epoch {epoch + 1}, skipping first {epoch_start_step} steps.")


        for i, batch_data in enumerate(data_loader):
            step = i + 1 # Current step (1-based) within this epoch

            # --- Skip steps if resuming mid-epoch ---
            if step <= epoch_start_step:
                continue
            # --- End Skip steps ---


            # Handle potential None batches from collate_fn if filtering occurred
            if batch_data is None or batch_data[0] is None:
                print(f"Skipping empty/invalid batch at step {step}")
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
                    print(f"Batch became empty after filtering short captions at step {step}. Skipping.")
                    continue
                print(f"Filtered {sum(~valid_indices)} short captions at step {step}")


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

            # --- Changed: Log uses 1-based epoch and step ---
            print(f'E[{epoch+1}/{config.num_epochs}] S[{step}/{total_steps}] Loss: {loss.item():.4f}') # Optional: Comment out if too verbose
            # --- End Changed Log ---


            # Log training status periodically (keep existing log for less frequent, more detailed info)
            if step % config.log_step == 0:
                perplexity = torch.exp(loss).item() if loss.item() < 100 else float('inf') # Avoid overflow
                # --- Changed: Log uses 1-based epoch and step ---
                print(f'--- Log Step --- Epoch [{epoch+1}/{config.num_epochs}], Step [{step}/{total_steps}], '
                      f'Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}')

            # Save model checkpoints
            if step % config.save_step == 0:
                # --- Changed: Save optimizer state as well ---
                epoch_num = epoch + 1 # Use 1-based epoch for filename
                encoder_path = os.path.join(config.model_path, f'encoder-{epoch_num}-{step}.ckpt')
                decoder_path = os.path.join(config.model_path, f'decoder-{epoch_num}-{step}.ckpt')
                optimizer_path = os.path.join(config.model_path, f'optimizer-{epoch_num}-{step}.ckpt') # Optimizer path
                torch.save(encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                torch.save(optimizer.state_dict(), optimizer_path) # Save optimizer
                print(f"--- Checkpoint Saved --- Epoch {epoch_num}, Step {step}. Models and Optimizer saved.")
                # --- End Changed ---

        # Optional: Add end-of-epoch saving or validation here
        # Reset start_step for the next epoch if we finished the first resumed epoch
        if epoch == start_epoch and checkpoint_loaded:
            start_step = 0 # Subsequent epochs start from step 0
            checkpoint_loaded = False # Ensure this only affects the very first epoch after loading

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

    # --- Changed: Save final optimizer state ---
    encoder_path = os.path.join(config.model_path, 'encoder-final.ckpt')
    decoder_path = os.path.join(config.model_path, 'decoder-final.ckpt')
    optimizer_path = os.path.join(config.model_path, 'optimizer-final.ckpt') # Final optimizer path
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    torch.save(optimizer.state_dict(), optimizer_path) # Save final optimizer
    print(f"Saved final models and optimizer to {encoder_path}, {decoder_path}, and {optimizer_path}")
    # --- End Changed ---


if __name__ == '__main__':
    # --- Changed: Use argparse ---
    parser = argparse.ArgumentParser(description='Train Show and Tell Model')
    # Add arguments for paths, hyperparameters etc. based on Config class
    parser.add_argument('--model_path', type=str, default=Config.model_path, help='Path for saving trained models')
    parser.add_argument('--log_path', type=str, default=Config.log_path, help='Path for saving logs')
    parser.add_argument('--data_path', type=str, default=Config.data_path, help='Base path containing annotations dir')
    parser.add_argument('--image_dir_relative', type=str, default=Config.image_dir_relative_to_data_path, help='Relative path to images from data_path')
    parser.add_argument('--caption_file', type=str, default=Config.caption_file, help='Relative path to annotation json within data_path')
    parser.add_argument('--vocab_path', type=str, default=Config.vocab_path, help='Path to load vocabulary')
    parser.add_argument('--embed_size', type=int, default=Config.embed_size, help='Dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=Config.hidden_size, help='Dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=Config.num_layers, help='Number of layers in lstm')
    parser.add_argument('--num_epochs', type=int, default=Config.num_epochs, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=Config.batch_size, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=Config.learning_rate, help='Learning rate')
    parser.add_argument('--log_step', type=int, default=Config.log_step, help='Step frequency for logging')
    parser.add_argument('--save_step', type=int, default=Config.save_step, help='Step frequency for saving checkpoints')
    parser.add_argument('--num_workers', type=int, default=Config.num_workers, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help="Path prefix to checkpoint to resume from (e.g., 'models/encoder-1-1000') or 'latest'")

    args = parser.parse_args()

    # Update config based on args
    config = Config() # Create instance
    # Update attributes from args
    config.model_path = args.model_path
    config.log_path = args.log_path
    config.data_path = args.data_path
    config.image_dir_relative_to_data_path = args.image_dir_relative
    config.caption_file = args.caption_file
    config.vocab_path = args.vocab_path
    config.embed_size = args.embed_size
    config.hidden_size = args.hidden_size
    config.num_layers = args.num_layers
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.log_step = args.log_step
    config.save_step = args.save_step
    config.num_workers = args.num_workers
    config.resume = args.resume # Store resume argument in config

    main(config)
    # --- End Changed --- 