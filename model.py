import torch
import re
import pickle
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

with open("./vocab.pkl", "rb") as Embeddings:
    vocab = pickle.load(Embeddings)

class ImprovedRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.4):
        super(ImprovedRNNModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(dropout * 0.5)

        # Multiple RNN layers with dropout
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Dropout after RNN
        self.rnn_dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        # Xavier/Glorot initialization for better gradient flow
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # For linear and RNN weights
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        # Embedding with dropout
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)

        # RNN processing
        rnn_out, hidden = self.rnn(embedded)

        # Use last hidden state
        last_hidden = hidden[-1]  # Take last layer's hidden state
        last_hidden = self.rnn_dropout(last_hidden)

        # Fully connected layers
        out = torch.relu(self.fc1(last_hidden))
        out = self.fc1_dropout(out)
        out = self.fc2(out)

        return out

# Model parameters
vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 128
output_dim = 2
num_layers = 1
dropout = 0.4

model = ImprovedRNNModel(vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout)

model.load_state_dict(torch.load("./sentiment_model.pth", map_location=torch.device("cpu")))

def predict(tweet):
    model.eval()
    max_len = 20
    # Clean the input tweet (same as your preprocessing)
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    # Encode function (same as training)
    def encode_tokens(tokens, vocab, max_len=20):
        encoded = [vocab.get(word, vocab['<UNK>']) for word in tokens]
        if len(encoded) < max_len:
            encoded += [vocab['<PAD>']] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        return encoded

    # Apply cleaning and tokenization
    cleaned = clean_text(tweet)
    tokens = word_tokenize(cleaned)
    encoded = encode_tokens(tokens, vocab, max_len)
    
    # Convert to tensor
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # Map prediction to label
    label_map = {0: 'negative', 1: 'positive'}  # adjust if your label encoder was different
    return label_map[predicted_class]
