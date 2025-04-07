import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import json
import nltk # For tokenization during data loading

# Assuming build_vocab.py is in the same directory or vocabulary.py exists
# If Vocabulary class is in build_vocab.py, you might need to move it
# to a separate file (e.g., vocabulary.py) to avoid running build_vocab logic
# when importing Vocabulary. For now, let's assume it's accessible.
from build_vocab import Vocabulary


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json_path, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root (string): Directory with all the images.
            json_path (string): Path to the COCO annotation json file.
            vocab (Vocabulary object): Vocabulary wrapper.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.vocab = vocab
        self.transform = transform
        # Load annotations
        try:
            with open(json_path, 'r') as f:
                self.annotations = json.load(f)['annotations']
            # Create a mapping from image_id to filename (assuming COCO format)
            # This might need adjustment based on your specific COCO version/structure
            # E.g., for train2017, filenames are like 000000xxxxxx.jpg
            # You might need a separate mapping if image filenames aren't directly derivable
            # from image_id or aren't stored conveniently.
            # A common approach is to load the images JSON as well.
            # Let's assume image filenames are directly related to image_id for simplicity.
            # This is often NOT the case, so you might need to load the 'images' part
            # of the COCO json to get file names from image ids.
            # Example (if you have images info):
            # with open(images_json_path, 'r') as f:
            #     images_info = json.load(f)['images']
            # self.imgid_to_filename = {img['id']: img['file_name'] for img in images_info}
            print(f"Loaded {len(self.annotations)} annotations from {json_path}")
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {json_path}")
            raise
        except KeyError:
             print(f"Error: 'annotations' key not found in {json_path}.")
             raise
        except Exception as e:
            print(f"Error loading annotations: {e}")
            raise

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        ann = self.annotations[index]
        caption = ann['caption']
        img_id = ann['image_id']

        # Construct image path (Modify this based on your actual COCO file structure)
        # Example for COCO 2017: filenames are zero-padded 12-digit numbers
        # This assumes 'root' points to the directory containing these images (e.g., 'train2017/')
        img_filename = f"{img_id:012d}.jpg"
        path = os.path.join(self.root, img_filename)

        try:
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {path}. Skipping index {index}.")
            # Handle missing image: return None or raise error, or try next item
            # Returning None requires careful handling in collate_fn or training loop
            # A simple fix is to return the next valid item (less efficient)
            # Or pre-filter annotations for existing images
            return self.__getitem__((index + 1) % len(self)) # Simple retry next

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_ids = []
        caption_ids.append(vocab('<start>'))
        caption_ids.extend([vocab(token) for token in tokens])
        caption_ids.append(vocab('<end>'))
        target = torch.Tensor(caption_ids).long() # Use LongTensor for indices
        return image, target

    def __len__(self):
        return len(self.annotations)


# Define PadCollate as a top-level class
class PadCollate:
    """
    A callable class that takes a batch of data and pads the captions.
    It stores the vocabulary to access the pad index.
    """
    def __init__(self, vocab):
        """
        Args:
            vocab (Vocabulary): Vocabulary object containing word indices.
        """
        self.vocab = vocab
        self.pad_idx = vocab('<pad>') # Store pad index for efficiency

    def __call__(self, data):
        """Creates mini-batch tensors from the list of tuples (image, caption).
        Args:
            data: list of tuple (image, caption).
        Returns:
            images: torch tensor of shape (batch_size, 3, H, W).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: torch tensor of shape (batch_size); valid length for each padded caption.
        """
        # Filter out None entries
        data = [item for item in data if item is not None]
        if not data:
            return None, None, None

        # Sort a data list by caption length (descending order).
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        lengths = [len(cap) for cap in captions]
        targets = torch.full((len(captions), max(lengths)), self.pad_idx, dtype=torch.long) # Use stored pad_idx

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        # Convert lengths to tensor
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        return images, targets, lengths_tensor


def get_loader(root, json_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for COCO dataset."""

    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json_path=json_path,
                       vocab=vocab,
                       transform=transform)

    # Instantiate the PadCollate class
    collate_instance = PadCollate(vocab)

    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_instance, # Pass the instance here
                                              pin_memory=True if torch.cuda.is_available() else False)
    return data_loader

# Example usage (for testing this file independently)
if __name__ == '__main__':
    # Example Usage Parameters (replace with your actual paths/settings)
    vocab_path = './vocab.pkl'
    coco_root = './train2017/train2017' # Directory containing COCO training images
    annotations_path = './annotations/captions_train2017.json'
    batch_size = 4 # Small batch for testing

    # Define a transform (should match the one in train.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary
    try:
        with open(vocab_path, 'rb') as f:
            vocab_main = pickle.load(f)
        print("Vocabulary loaded.")
    except FileNotFoundError:
        print(f"Vocabulary file not found at {vocab_path}. Run build_vocab.py first.")
        exit()

    # Get the data loader
    loader = get_loader(coco_root, annotations_path, vocab_main, transform, batch_size, shuffle=True, num_workers=0) # Use num_workers=0 for easier debugging

    print(f"DataLoader created. Vocabulary size: {len(vocab_main)}")

    # Test iterating through one batch
    try:
        print("Testing data loader iteration...")
        images, captions, lengths = next(iter(loader))

        if images is None:
             print("Data loader returned an empty first batch (possibly due to filtering).")
        else:
            print("Successfully retrieved one batch.")
            print("Images shape:", images.shape)
            print("Captions shape:", captions.shape)
            print("Lengths:", lengths)
            print("Caption example (indices):", captions[0])
            # Decode first caption example using the loaded vocab
            caption_text = [vocab_main.idx2word[idx.item()] for idx in captions[0] if idx.item() in vocab_main.idx2word]
            print("Caption example (decoded):", ' '.join(caption_text))

    except StopIteration:
        print("Data loader is empty or could not retrieve a batch.")
    except Exception as e:
        print(f"An error occurred during data loader testing: {e}")
        import traceback
        traceback.print_exc() 