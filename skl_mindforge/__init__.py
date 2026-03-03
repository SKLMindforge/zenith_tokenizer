import os
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

class ZenithTokenizer:
    def __init__(self, model_filename="private_vocab_40k.json"):
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, model_filename)
        
        if not os.path.exists(model_path):
            model_path = model_filename 
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_filename}")
            
        self.tokenizer = Tokenizer.from_file(model_path)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", 1), ("</s>", 2)],
        )
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        """
        VERSION 0.1.6: The Byte-Level Recovery Decoder.
        Uses the 'latin-1' byte-shuffle trick to perfectly reconstruct 
        complex Math and Unicode symbols.
        """
        # 1. Extract raw tokens from the vocabulary
        tokens = [self.tokenizer.id_to_token(i) for i in ids]
        
        # 2. Filter Specials
        if skip_special_tokens:
            special = {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}
            tokens = [t for t in tokens if t not in special]

        # 3. REVERSE THE MOJIBAKE
        # Most BPE tokenizers represent raw bytes as latin-1 characters.
        # We join them into one string, encode to latin-1 to get the raw bytes,
        # then decode those bytes as UTF-8 to get the real symbols (ℏ, ∇, etc.)
        try:
            raw_serialized = "".join(tokens)
            # This is the industry-standard fix for GPT-style byte-level BPE
            clean = raw_serialized.encode('latin-1').decode('utf-8', errors='ignore')
        except Exception:
            # Emergency fallback if byte-decoding fails
            clean = "".join(tokens).replace('Ġ', ' ').replace('Ċ', '\n')

        # 4. FINAL POLISH
        # Replace the 'Ġ' that didn't get caught in the byte-shuffle
        clean = clean.replace('Ġ', ' ').replace('Ċ', '\n')
        
        # Punctuation spacing correction
        clean = clean.replace(' ,', ',').replace(' .', '.').replace(' - ', '-')
        
        return clean.strip()

zenith_tokenizer = ZenithTokenizer
