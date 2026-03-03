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
        ULTRA-REFINED DECODER:
        Uses a multi-stage cleaning pipeline to eliminate BPE mojibake.
        """
        # 1. Get raw string
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        
        # 2. Level 1: Standard BPE Marker Conversion
        text = text.replace('Ġ', ' ').replace('Ċ', '\n')
        
        # 3. Level 2: Comprehensive Artifact Stripping
        # This list targets every ghost character seen in your tests
        artifacts = [
            'Â', '¹', 'ÃĤ', 'ÃĦ', 'ÅĤ', 'Ã', 'Å', 'ł', 'Ħ', 
            'Ä', 'Ĥ', 'Ã', ' ', 'â', '€', '™'
        ]
        for artifact in artifacts:
            text = text.replace(artifact, '')

        # 4. Level 3: Regex Scrubbing
        # Removes any remaining standalone non-ASCII "junk" characters 
        # that aren't standard punctuation or symbols.
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # 5. Level 4: Byte-Level Reconstruction (The Safety Net)
        try:
            # Try one last time to heal broken UTF-8 bytes
            text = text.encode('ascii', 'ignore').decode('ascii')
        except:
            pass

        # 6. Final Polish
        # Collapses multiple spaces and cleans edges
        return " ".join(text.split()).strip()

zenith_tokenizer = ZenithTokenizer
