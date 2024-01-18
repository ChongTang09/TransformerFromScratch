import torch
import torch.nn as nn
import math

from collections import Counter

class ConditionalLayerNormalization(nn.Module):
    def __init__(self, features: int, embedding_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Original scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Original bias parameter

        # # LangAdapter is an additional linear layer to process language embeddings
        # self.lang_adapter = nn.Linear(embedding_size, embedding_size)

        # Linear layers for conditional parameters based on processed language embeddings
        self.lang_scale = nn.Linear(embedding_size, features)
        self.lang_bias = nn.Linear(embedding_size, features)

    def forward(self, x, lang_embedding):
        # x: (batch, seq_len, hidden_size)
        # lang_embedding: (batch, embedding_size)
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # Normalize
        normalized_x = (x - mean) / (std + self.eps)

        # # Process language embeddings through LangAdapter
        # lang_embedding = self.lang_adapter(lang_embedding)

        # Obtain conditional scale and shift parameters
        cond_scale = self.lang_scale(lang_embedding).unsqueeze(1)  # (batch, 1, hidden_size)
        cond_bias = self.lang_bias(lang_embedding).unsqueeze(1)  # (batch, 1, hidden_size)

        # Apply conditional normalization
        return self.alpha * cond_scale * normalized_x + self.bias + cond_bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class DynamicLanguageAdaptiveInputEmbeddings(nn.Module):

    def __init__(self, d_model: int, initial_vocab_size: int = 1000, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.current_vocab_size = initial_vocab_size
        self.base_embedding = nn.Embedding(initial_vocab_size, d_model)
        self.lang_adapters = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)
        self.is_first_language = True
        self.processed_languages = set()  # Track the languages that have been processed

    def update_for_new_language(self, lang_id: int, new_vocab_size: int):
        old_vocab_size = self.current_vocab_size  # Store the old vocab size

        if new_vocab_size > self.current_vocab_size:
            self._expand_vocab_size(new_vocab_size)

            if not self.is_first_language:
                # Freeze only the old part of the base embeddings
                self._freeze_weights(self.base_embedding, end_index=old_vocab_size)

        # Freeze the previous language adapters
        for old_lang_id in self.processed_languages:
            self._freeze_weights(self.lang_adapters[f"lang_{old_lang_id}"])

        self.lang_adapters[f"lang_{lang_id}"] = nn.Linear(self.d_model, self.d_model)
        self.processed_languages.add(lang_id)

        self.is_first_language = False

    def _expand_vocab_size(self, new_vocab_size: int):
        new_embedding = nn.Embedding(new_vocab_size, self.d_model)
        with torch.no_grad():
            new_embedding.weight[:self.current_vocab_size] = self.base_embedding.weight
        self.base_embedding = new_embedding
        self.current_vocab_size = new_vocab_size

    def _freeze_weights(self, module, end_index=None):
        for param in module.parameters():
            if end_index is not None:
                param.data[:end_index].requires_grad = False
            else:
                param.requires_grad = False

    def forward(self, x, lang_id: int):
        base_embeds = self.dropout(self.base_embedding(x))
        lang_embeds = self.lang_adapters[f"lang_{lang_id}"](base_embeds)
        return lang_embeds * math.sqrt(self.d_model)

class LanguageEmbedding(nn.Module):

    def __init__(self, num_languages: int, embedding_size: int):
        super(LanguageEmbedding, self).__init__()
        self.lang_type_embedding = nn.Embedding(num_languages, embedding_size)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"

    def forward(self, x, lang_type_ids):
        # lang_type_ids: Tensor of shape [batch_size] with language type IDs
        lang_type_tensor = torch.full((x.size(0),), lang_type_ids, dtype=torch.long).to(self.device)
        return self.lang_type_embedding(lang_type_tensor)  # [batch_size, embedding_dim]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
    def __init__(self, features: int, embedding_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Use ConditionalLayerNormalization instead of LayerNormalization
        self.norm = ConditionalLayerNormalization(features, embedding_size)

    def forward(self, x, sublayer, lang_embedding):
        # Pass the language embedding to the conditional layer normalization
        return x + self.dropout(sublayer(self.norm(x, lang_embedding)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, embedding_size: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Note the inclusion of embedding_size in the ResidualConnection
        self.residual_connections = nn.ModuleList([ResidualConnection(features, embedding_size, dropout) for _ in range(2)])

    def forward(self, x, src_mask, lang_embedding):
        # Include lang_embedding in the residual connections
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask), lang_embedding)
        x = self.residual_connections[1](x, self.feed_forward_block, lang_embedding)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, embedding_size: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # Use ConditionalLayerNormalization instead of LayerNormalization
        self.norm = ConditionalLayerNormalization(features, embedding_size)

    def forward(self, x, mask, lang_embedding):
        for layer in self.layers:
            # Pass the language embedding to each layer
            x = layer(x, mask, lang_embedding)
        # Apply conditional normalization at the end
        return self.norm(x, lang_embedding)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, embedding_size: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Initialize ResidualConnection with embedding_size for ConditionalLayerNormalization
        self.residual_connections = nn.ModuleList([ResidualConnection(features, embedding_size, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask, lang_embedding):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask), lang_embedding)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask), lang_embedding)
        x = self.residual_connections[2](x, self.feed_forward_block, lang_embedding)
        return x

    
class Decoder(nn.Module):

    def __init__(self, features: int, embedding_size: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # Use ConditionalLayerNormalization instead of LayerNormalization
        self.norm = ConditionalLayerNormalization(features, embedding_size)

    def forward(self, x, encoder_output, src_mask, tgt_mask, lang_embedding):
        for layer in self.layers:
            # Pass the language embedding to each layer
            x = layer(x, encoder_output, src_mask, tgt_mask, lang_embedding)
        # Apply conditional normalization at the end
        return self.norm(x, lang_embedding)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: DynamicLanguageAdaptiveInputEmbeddings, tgt_embed: DynamicLanguageAdaptiveInputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, src_lang_embed: LanguageEmbedding, tgt_lang_embed: LanguageEmbedding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.src_lang_embed = src_lang_embed
        self.tgt_lang_embed = tgt_lang_embed
        self.projection_layer = projection_layer

    def encode(self, src, src_mask, lang_type_ids):
        src = self.src_embed(src, lang_type_ids)
        src = self.src_pos(src)
        lang_embedding = self.src_lang_embed(src, lang_type_ids)
        return self.encoder(src, src_mask, lang_embedding)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask, lang_type_ids):
        tgt = self.tgt_embed(tgt, lang_type_ids)
        tgt = self.tgt_pos(tgt)
        lang_embedding = self.tgt_lang_embed(tgt, lang_type_ids)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask, lang_embedding)
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, num_src_languages: int, num_tgt_languages: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = DynamicLanguageAdaptiveInputEmbeddings(d_model)
    src_embed.update_for_new_language(lang_id=0, new_vocab_size=src_vocab_size)
    tgt_embed = DynamicLanguageAdaptiveInputEmbeddings(d_model)
    tgt_embed.update_for_new_language(lang_id=1, new_vocab_size=tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create language embedding layers
    src_lang_embed = LanguageEmbedding(num_src_languages, d_model)
    tgt_lang_embed = LanguageEmbedding(num_tgt_languages, d_model)

    # Create the encoder and decoder blocks
    encoder_blocks = []
    decoder_blocks = []

    for _ in range(N):
        # Encoder block
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, d_model, encoder_self_attention_block, feed_forward_block, dropout) # added d_model for embedding size
        encoder_blocks.append(encoder_block)

        # Decoder block
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout) # added d_model for embedding size
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, d_model, nn.ModuleList(encoder_blocks)) # added d_model for embedding size
    decoder = Decoder(d_model, d_model, nn.ModuleList(decoder_blocks)) # added d_model for embedding size

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, src_lang_embed, tgt_lang_embed, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
