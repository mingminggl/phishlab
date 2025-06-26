import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

AI_LIKE_PHRASES = [
    "as an ai language model", "i cannot provide", "in conclusion",
    "it's important to note", "as previously mentioned"
]

# Load GPT-2 model and tokenizer once
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

def calculate_perplexity(text):
    encodings = gpt2_tokenizer(text, return_tensors="pt")
    max_length = gpt2_model.config.n_positions
    stride = 512
    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * input_ids.size(1)
        lls.append(log_likelihood)
    ppl = torch.exp(torch.stack(lls).sum() / encodings.input_ids.size(1))
    return ppl.item()

def burstiness(text):
    sentences = [s for s in text.replace('\n', ' ').split('.') if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) < 2:
        return 0
    return np.std(lengths) / np.mean(lengths)

def detect_ai_signature(text, perplexity_threshold=40, burstiness_threshold=0.3):
    if not isinstance(text, str):
        text = str(text)
    text_lower = text.lower()
    # Heuristic phrase check
    for phrase in AI_LIKE_PHRASES:
        if phrase in text_lower:
            return True
    # Perplexity check
    try:
        ppl = calculate_perplexity(text)
        if ppl < perplexity_threshold:
            return True
    except Exception:
        pass
    # Burstiness check
    b = burstiness(text)
    if b < burstiness_threshold:
        return True
    return False