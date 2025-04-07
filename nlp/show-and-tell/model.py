import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

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
