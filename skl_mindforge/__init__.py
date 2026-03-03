import os
import re
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
        VERSION 0.1.4: The 'Glue' Decoder.
        Reassembles sub-tokens (Ch + ocol + ater + ie) into whole words.
        """
        # 1. Convert IDs to their raw strings from the vocab (no auto-spaces)
        tokens = [self.tokenizer.id_to_token(i) for i in ids]
        
        # 2. Filter out special tokens
        if skip_special_tokens:
            special = {"<s>", "</s>", "<pad>", "<unk>", "<mask()", "[CLS]", "[SEP]", "[PAD]"}
            tokens = [t for t in tokens if t not in special]

        # 3. GLUE: Join everything with NO spaces first
        raw_text = "".join(tokens)

        # 4. CONVERT BPE MARKERS: 'Ġ' becomes a space, 'Ċ' becomes newline
        clean = raw_text.replace('Ġ', ' ').replace('Ċ', '\n')
        
        # 5. SURGICAL NOISE REMOVAL
        artifacts = ['Â', '¹', 'ÃĤ', 'ÃĦ', 'ÅĤ', 'Ã', 'Å', 'ł', 'Ħ', 'Ä', 'Ĥ']
        for artifact in artifacts:
            clean = clean.replace(artifact, '')

        # 6. SCORCHED EARTH REGEX: Strip remaining non-ASCII junk
        clean = re.sub(r'[^\x00-\x7F]+', '', clean)

        # 7. PUNCTUATION POLISH: Fix spacing around symbols
        # Reverses the "1940 , " -> "1940, " artifact
        clean = clean.replace(' ,', ',').replace(' .', '.').replace(' ( ', ' (').replace(' )', ')')
        clean = clean.replace(' - ', '-') # Fixes high-quality
        
        return clean.strip()

zenith_tokenizer = ZenithTokenizer
