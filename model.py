import torch
import torch.nn
import math 

class InputEmbeddings(nn.Module): 
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x): 
        return self.embedding(x) * math.sqrt(d_model)

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:  
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #vector of shape(1, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        #apply sine to even positions and cosine to odd positions 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsequeeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe) #save tensor when file is closed
        
    
    def forward(self, x): 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)

        



