import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F # Needed for log_softmax
import heapq # Needed for beam search priority queue

class EncoderCNN(nn.Module):
    """
    CNN Encoder using a pretrained ResNet.
    It extracts feature vectors from input images.
    """
    def __init__(self, embed_size):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # Remove the final fully connected layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        # Freeze ResNet layers initially (optional, can be fine-tuned later)
        for param in self.resnet.parameters():
            param.requires_grad_(False)

    def forward(self, images):
        """Extract feature vectors from input images."""
        # Expected input shape: (batch_size, 3, image_size, image_size) e.g., (B, 3, 224, 224)
        with torch.no_grad(): # Or enable gradients if fine-tuning
             features = self.resnet(images) # (B, C, 1, 1) e.g., (B, 2048, 1, 1)
        features = features.reshape(features.size(0), -1) # Flatten: (B, C) e.g., (B, 2048)
        features = self.bn(self.linear(features)) # Project to embed_size: (B, embed_size)
        return features


class DecoderRNN(nn.Module):
    """
    RNN Decoder (LSTM) which generates captions word by word.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        # features: (batch_size, embed_size)
        # captions: (batch_size, max_seq_length)
        # lengths: (batch_size) list of sequence lengths for each caption

        embeddings = self.embed(captions) # (batch_size, max_seq_length, embed_size)

        # Prepend image features to caption embeddings
        # Unsqueeze features to make it (batch_size, 1, embed_size) to concatenate
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # (batch_size, max_seq_length+1, embed_size)

        # Pack sequence to handle variable lengths efficiently
        # Note: lengths+1 because we added the image feature as the first input
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)

        hiddens, _ = self.lstm(packed) # Packed output

        # Unpack sequence (optional, depending on how you use the output)
        # hiddens_unpacked, _ = pad_packed_sequence(hiddens, batch_first=True)

        # Pass LSTM outputs through the final linear layer
        outputs = self.linear(hiddens[0]) # hiddens[0] contains the packed hidden states
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        # features: (batch_size, embed_size) -> expected B=1 for sampling typically
        sampled_ids = []
        # features shape needs to be (1, 1, embed_size) for LSTM input
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (1, 1, hidden_size), states: (num_layers, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (1, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (1, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (1, 1, embed_size)
            # Stop if <end> token is generated (assuming <end> has a specific index, e.g., 1)
            # You'll need to know the index of your <end> token from your vocabulary
            # if predicted.item() == END_TOKEN_IDX:
            #    break
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (1, max_seq_length)
        return sampled_ids

    # --- Added Beam Search Method ---
    def beam_search_sample(self, features, vocab, beam_width=5):
        """Generate captions for given image features using beam search."""
        # Get device from features
        device = features.device
        batch_size = features.size(0)
        if batch_size != 1:
            raise ValueError("Beam search currently supports batch_size=1 only.")

        # Get special token indices
        start_idx = vocab(vocab.word2idx['<start>'])
        end_idx = vocab(vocab.word2idx['<end>'])

        # Initialize
        k = beam_width
        completed_beams = [] # List to store completed sequences [(score, sequence)]
        active_beams = [(0.0, [start_idx], None)] # List to store active beams [(log_prob_score, sequence, lstm_state)]

        # Start with image features
        inputs = features.unsqueeze(1) # (1, 1, embed_size)

        # Run beam search step by step
        for _ in range(self.max_seg_length):
            next_beam_candidates = []

            # Check if we have enough completed beams
            if len(completed_beams) >= k:
                break

            new_active_beams = []
            for log_prob, seq, current_states in active_beams:
                last_word_idx = seq[-1]

                # If the last word is <end>, add to completed and continue
                if last_word_idx == end_idx:
                    completed_beams.append((log_prob, seq))
                    # Prune completed beams if exceeding k (keep top k)
                    if len(completed_beams) > k:
                         completed_beams = heapq.nlargest(k, completed_beams, key=lambda item: item[0])
                    continue # Don't expand completed sequences
                
                # Prepare input for LSTM step
                current_input_word = torch.tensor([last_word_idx], dtype=torch.long).to(device)
                lstm_input = self.embed(current_input_word).unsqueeze(1) # (1, 1, embed_size)

                # LSTM forward step
                hiddens, next_states = self.lstm(lstm_input, current_states) # hiddens: (1, 1, hidden_size)
                outputs = self.linear(hiddens.squeeze(1)) # outputs: (1, vocab_size)
                log_probs = F.log_softmax(outputs, dim=1) # (1, vocab_size)

                # Get top k candidates for the next word
                top_log_probs, top_indices = log_probs.topk(k, dim=1) # (1, k), (1, k)

                # Add new candidates to consider
                for i in range(k):
                    next_word_idx = top_indices[0, i].item()
                    next_log_prob = top_log_probs[0, i].item()
                    new_score = log_prob + next_log_prob
                    new_seq = seq + [next_word_idx]
                    next_beam_candidates.append((new_score, new_seq, next_states))

            # If no candidates generated, break (shouldn't happen normally)
            if not next_beam_candidates:
                 break

            # Keep the top k overall candidates from all expanded beams
            active_beams = heapq.nlargest(k - len(completed_beams), next_beam_candidates, key=lambda item: item[0]) # Keep k beams total (active + completed)

        # If no beams completed, use the best active beam
        if not completed_beams:
             completed_beams = [(score, seq) for score, seq, _ in active_beams]

        # Sort completed beams by score (highest first)
        completed_beams.sort(key=lambda item: item[0], reverse=True)

        # Return the sequence of the best beam (highest score)
        best_score, best_seq = completed_beams[0]
        #print(f"Beam Search Best Score: {best_score}") # Optional: print score
        return best_seq
    # --- End Beam Search Method ---
