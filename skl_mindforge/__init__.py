import os
import re
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

class ZenithTokenizer:
    def __init__(self, model_filename="private_vocab_40k.json"):
        # 1. Locate the file inside the package folder
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, model_filename)
        
        if not os.path.exists(model_path):
            # Fallback for different environments
            model_path = model_filename 
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_filename}")
            
        # 2. Load the core Rust-based tokenizer
        self.tokenizer = Tokenizer.from_file(model_path)
        
        # 3. Post-processor for the Chat/Assistant format
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", 1), ("</s>", 2)],
        )
        
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        """Converts raw text into a list of token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Converts IDs back to text and cleans Zenith-specific BPE artifacts.
        Handles the Ġ, ¹, and byte-level mojibake (Â, Ä ł).
        """
        # 1. Get the raw string from the library
        # This will contain the 'Ġ' and 'Â' symbols initially
        raw_output = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        
        # 2. Convert BPE space/newline markers to real whitespace
        clean = raw_output.replace('Ġ', ' ').replace('Ċ', '\n')
        
        # 3. Strip the start-token artifact
        clean = clean.replace('¹', '')
        
        # 4. FIX THE MOJIBAKE (The Â, Ä, ł stuff)
        # We use 'latin-1' to get raw bytes, then 'utf-8' to rebuild the characters correctly
        try:
            # This is the "Magic" step that fixes the 'ĠÄ ł' mess
            clean = clean.encode('latin-1').decode('utf-8', errors='ignore')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback if the byte-shuffle fails
            clean = re.sub(r'[ÂÄł¹]', '', clean)
            
        # 5. Final Polish: remove redundant whitespace and strip edges
        return " ".join(clean.split()).strip()

# Alias for compatibility
zenith_tokenizer = ZenithTokenizer
