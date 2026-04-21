import torch
import torch.nn.functional as F
import sentencepiece as spm
import os
import argparse
from train_gpt_recurrent import GPT, Hyperparameters

class GPT_with_inference(GPT):
    def forward(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        
        # Loop based on num_loops trait from train_gpt_recurrent
        for _ in range(self.num_loops):
            skips = []
            # First half stores skips; second half reuses them in reverse order.
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)
        
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50, device='cuda'):
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Autoregressive generation
    for _ in range(max_new_tokens):
        # Cropping context if it exceeds seq_len (though RNN architecture handles this, 
        # the pos_emb in the base model might be fixed)
        x_cond = x if x.size(1) <= 1024 else x[:, -1024:]
        
        # Forward pass
        with torch.no_grad():
            logits = model(x_cond)
            # Only take the last position
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, next_token), dim=1)
            
            # Check for EOS or if we should stop
            if next_token.item() == tokenizer.eos_id():
                break
                
    return tokenizer.decode(x[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="The start of the text")
    parser.add_argument("--max_tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Creativity (lower is more focused)")
    parser.add_argument("--top_k", type=int, default=40, help="Limit to top K most likely tokens")
    parser.add_argument("--model_path", type=str, default="final_model.pt", help="Path to weights")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    # 2. Hyperparameters (Must match train_gpt settings exactly)
    hp = Hyperparameters()
    
    # 3. Initialize Model
    # Passing hp.num_loops (default 2 in train_gpt_recurrent)
    model = GPT_with_inference(
        hp.vocab_size, hp.num_layers, hp.model_dim, hp.num_heads, hp.num_kv_heads, 
        hp.mlp_mult, hp.tie_embeddings, hp.tied_embed_init_std, hp.logit_softcap, 
        hp.rope_base, hp.qk_gain_init, hp.num_loops
    ).to(device)
    
    # 4. Load Weights
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {args.model_path} not found. Running with uninitialized weights.")

    # 5. Generate
    print("\nWelcome to Recurrent GPT Inference!")
    print(f"Model Configuration: {hp.num_layers} layers x {hp.num_loops} loops")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            prompt = input("Prompt: ")
            if prompt.lower() in ['exit', 'quit']:
                break
            if not prompt.strip():
                continue
                
            print("\n--- Generating ---")
            output = generate(model, sp, prompt, max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k, device=device)
            print("\nResult:")
            print(output)
            print("------------------\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
