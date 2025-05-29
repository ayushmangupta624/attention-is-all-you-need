import torch
import torch.nn as nn
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


class Norm(nn.Module): 
    def __init__(self, eps: float = 10**-6): 
        super().__init__()
        self.epsilon = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplicative
        self.beta = nn.Parameter(torch.zeros(1)) #Additive

        
    def forward(self, x): 
        x = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.beta

class FeedForward(nn.Module): 
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: 
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x): 
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class MultiHead(nn.Module): 

    def __init__(self, h: int, d_model: int, dropout: float) -> None:
       super().__init__()
       assert d_model % h == 0, "d_model is not divisible by h"
       self.d_k = d_model // h
       self.W_Q = nn.Linear(d_model, d_model)
       self.W_K = nn.Linear(d_model, d_model)
       self.W_V = nn.Linear(d_model, d_model) 
       self.W_O = nn.Linear(d_model, d_model)
       self.dropout(nn.Dropout)

    @staticmethod
    def self_attention(query, key, value, mask: int, dropout: nn.Dropout): 
        #(batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
       
        d_k = key.shape[-1]
        attention_logits = (query @ key.transpose(2, 3)) / math.sqrt(d_k)
        if mask is not None: 
            attention_logits = attention_logits.masked_fill_(mask == 0, -1e9)
        attention_logits = attention_logits.softmax(dim=-1)
        if dropout is not None: 
            attention_logits = dropout(attention_logits)

        return (attention_logits @ value), attention_logits #return for visualization of logits


    def forward(self, q, k, v, mask): 
        Q = self.W_Q(q)
        K = self.W_Q(k)
        V = self.W_Q(v)

        Q = Q.view(Q.shape[0], Q.shape[1], h, self.d_k).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], h, self.d_k).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], h, self.d_k).transpose(1, 2)

        x, self.attention_logits = MultiHead.self_attention(Q, K, V, mask, self.dropout)
      
        #(batch, h, seq_len, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, h * self.d_k)
        return self.W_O(x)

class ResidualConnection(nn.Module): 
    def __init__(self, dropout: float) -> None: 
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Norm()

    def forward(self, x, prev_layer): 
        return x + self.dropout(self.norm(prev_layer(x)))

class EncoderBlock(nn.Module): 
    def __init__(self, multi_head: MultiHead, ff: FeedForward, dropout: float) -> None: 
        super().__init__()
        self.multi_head_attention = multi_head
        self.feed_forward = self.ff
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual[0](x, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual[1](x, self.feed_forward)
        return x 

class Encoder(nn.Module): 
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = Norm()

    def forward(self, x, mask): 
        for layer in layers: 
            x = layer(x, mask)
        return self.norm(x)   


class DecoderBlock(nn.Module): 
    def __init__(self, multi_head_attention: MultiHead, cross_attention: MultiHead, feed_forward: FeedForward, dropout: float) -> None: 
        self.multi_head_attention = multi_head_attention
        self.cross_attention = cross_attention 
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask): 
        x = self.residual[0](x, lambda x: self.multi_head_attention(x, x, x, tgt_mask))
        x = self.residual[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual[2](x, lambda x: self.feed_forward(x))
        return x 
        

class Decoder(nn.Module): 
    def __init__(self, layers: nn.ModuleList) -> None: 
        super().__init__()
        self.layers = layers
        self.norm = Norm()

    def forward(self, x, encoder_output, src_mask, tgt_mask): 
        for layer in self.layers: 
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module): 
    def __init__(self, d_model: int, vocab_size: int) -> None: 
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x): 
        return torch.log_softmax(self.projection(x), dim=-1)


class Transformer(nn.Module): 
    def __init__(self, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, encoder: Encoder, decoder: Decoder, projection: ProjectionLayer) -> None:
        super().__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection

    def encode(self, src, src_mask): 
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask): 
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)


    def project(self, x): 
        return self.projection(x)
    


def BuildTransformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer: 
    #Embedding layers 
    src_embed = InputEmbeddings(d_model, src_vocab_size) 
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    #Positional encoding layers 
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)


    encoders = []
    for _ in range(N): 
        encoder_self_attention_block = MultiHead(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoders.appent(encoder_block)

    
    decoders = []
    for _ in range(N): 
        decoder_self_attention_block = MultiHead(d_model, h, dropout)
        decoder_cross_attention_block = MultiHead(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoders.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoders))
    decoder = Decoder(nn.ModuleList(decoders))

    proj = Projection(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj)      

    #Initializing parameters 
    for p in transformer.parameters(): 
        if p.dim() > 1: 
            nn.init.xavier_uniform_(p)

    return transformer 





