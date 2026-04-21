import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import os
import io
import zlib
import argparse
import time

# -----------------------------
# ARCHITECTURE DEFINITION
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        # No learnable weight in this version

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x, seq_len):
        if seq_len != self._seq_len_cached:
            t = torch.arange(seq_len, device=x.device).float()
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=x.dtype, device=x.device), self._sin_cached.to(dtype=x.dtype, device=x.device)

def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(num_heads))
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        
        cos, sin = self.rotary(q, seqlen)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        q = q * self.q_gain[None, :, None, None]
        
        # Manual GQA broadcasting
        if self.num_heads != self.num_kv_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = nn.Linear(dim, mlp_mult * dim, bias=False)
        self.proj = nn.Linear(mlp_mult * dim, dim, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))

    def forward(self, x, x0):
        # x0 is the initial embedding projection
        mix = self.resid_mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult, logit_softcap):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim))
        
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(model_dim)

    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
            
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i][None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
            
        x = self.final_norm(x)
        # Tied embeddings
        logits_proj = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits

# -----------------------------
# DEQUANTIZATION LOGIC
# -----------------------------

def dequantize_state_dict_int8(obj: dict) -> dict:
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

def load_quantized_model(path, model, device='cpu'):
    print(f"Loading and dequantizing model from {path}...")
    with open(path, "rb") as f:
        quant_blob = f.read()
    
    # Decompress and load
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu", weights_only=False)
    
    # Dequantize
    state_dict = dequantize_state_dict_int8(quant_state)
    
    # Load into model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

# -----------------------------
# INFERENCE LOGIC
# -----------------------------

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50, device='cuda'):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= 1024 else x[:, -1024:]
        
        with torch.no_grad():
            logits = model(x_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            
            if next_token.item() == tokenizer.eos_id():
                break
                
    return tokenizer.decode(x[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Humanity's greatest achievement is", help="The start of the text")
    parser.add_argument("--max_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Creativity")
    parser.add_argument("--top_k", type=int, default=40, help="Limit to top K")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Configuration (Must match saved model)
    vocab_size = 1024
    num_layers = 9
    model_dim = 512
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 2
    logit_softcap = 30.0

    # 1. Initialize Model
    model = GPT(vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult, logit_softcap)
    if device == 'cuda':
        model = model.to(dtype=torch.bfloat16)

    # 2. Load and Dequantize
    model_path = "final_model.int8.ptz"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    
    # Update load_quantized_model logic internally or just load and then cast
    load_quantized_model(model_path, model, device=device)
    if device == 'cuda':
        model = model.to(dtype=torch.bfloat16)

    # 3. Tokenizer
    tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)

    # 4. Generate
    print(f"\nPrompt: {args.prompt}")
    print("\n--- Generating ---")
    start_time = time.time()
    result = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k, device=device)
    end_time = time.time()
    
    print(f"\nResult:\n{result}")
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    print("-" * 50)

    # Interactive loop
    print("\nEntering interactive mode. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("\nPrompt: ")
            if prompt.lower() == 'exit':
                break
            if not prompt.strip():
                continue
            
            output = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k, device=device)
            print(f"\nGenerated:\n{output}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
