import torch
import torchvision.transforms as T
from PIL import Image
import argparse
import pickle
import os

# Assuming model definitions are in 'model.py' and vocab in 'build_vocab.py'
from model import EncoderCNN, DecoderRNN
from build_vocab import Vocabulary

# Device configuration (same as training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, transform=None):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS) # Resize before transform

    if transform is not None:
        image = transform(image).unsqueeze(0) # Add batch dimension
    return image

def generate_caption(encoder, decoder, image_tensor, vocab, beam_width, max_length=50):
    """Generate a caption for an image using the decoder's beam search sample method."""
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode

    image_tensor = image_tensor.to(device)

    with torch.no_grad(): # Inference doesn't need gradients
        features = encoder(image_tensor) # (1, embed_size)

        # --- Use decoder.beam_search_sample method --- 
        if hasattr(decoder, 'beam_search_sample') and callable(getattr(decoder, 'beam_search_sample')):
            # Pass features, vocab, and beam_width
            # Note: The beam_search_sample method in model.py returns a list of indices
            sampled_ids = decoder.beam_search_sample(features, vocab, beam_width=beam_width)

        else:
             print("Error: Decoder object does not have a 'beam_search_sample' method. Cannot generate caption.")
             return "[Error: beam_search_sample method missing in DecoderRNN]"
        # --- End Use decoder.beam_search_sample method ---


    # Convert indices to words
    caption_words = [vocab.idx2word[idx] for idx in sampled_ids if idx in vocab.idx2word] # Add check for idx validity

    # Filter out special tokens for the final sentence
    caption = []
    for word in caption_words:
        if word == '<start>':
            continue
        # The sample method might or might not include <end>, handle it here
        if word == '<end>':
            break
        caption.append(word)

    return ' '.join(caption)


def main(args):
    # --- Setup ---
    # Image transformations (should match training transforms)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
    ])

    # Load vocabulary
    try:
        with open(args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {args.vocab_path}")
        vocab_size = len(vocab)
    except FileNotFoundError:
         print(f"Error: Vocabulary file not found at {args.vocab_path}.")
         return
    except Exception as e:
         print(f"Error loading vocabulary: {e}")
         return

    # --- Model Loading ---
    # Define model parameters (should match the trained model)
    # These could also be loaded from a config file saved during training
    embed_size = args.embed_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    # Initialize models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    print("Models initialized.")

    # Load trained weights
    try:
        if not os.path.exists(args.encoder_path):
             raise FileNotFoundError(f"Encoder checkpoint not found at {args.encoder_path}")
        if not os.path.exists(args.decoder_path):
             raise FileNotFoundError(f"Decoder checkpoint not found at {args.decoder_path}")

        print(f"Loading encoder weights from: {args.encoder_path}")
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        print(f"Loading decoder weights from: {args.decoder_path}")
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model weights: {e}")
        return
    except Exception as e:
        print(f"An error occurred loading model weights: {e}")
        import traceback
        traceback.print_exc()
        return


    # --- Inference ---
    # Load and preprocess the image
    try:
        image_tensor = load_image(args.image_path, transform)
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Generate the caption
    print(f"\nGenerating caption for: {args.image_path} (Beam Width: {args.beam_width})")
    caption = generate_caption(encoder, decoder, image_tensor, vocab, args.beam_width, args.max_length)

    # --- Output ---
    print("\nGenerated Caption:")
    print(caption)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate caption for an image using Show and Tell model.')

    # Input arguments
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='Path to saved vocabulary file.')

    # Model checkpoint arguments
    parser.add_argument('--encoder_path', type=str, required=True, help='Path to trained encoder model checkpoint (.ckpt)')
    parser.add_argument('--decoder_path', type=str, required=True, help='Path to trained decoder model checkpoint (.ckpt)')

    # Model hyperparameter arguments (must match trained model)
    parser.add_argument('--embed_size', type=int, default=256, help='Dimension of word embedding vectors.')
    parser.add_argument('--hidden_size', type=int, default=512, help='Dimension of LSTM hidden states.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in LSTM.')

    # Generation arguments
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of the generated caption.')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search decoding.')

    args = parser.parse_args()

    # Print arguments being used
    print("--- Inference Configuration ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("-----------------------------")


    main(args)
