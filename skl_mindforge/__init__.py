import os
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.processors import TemplateProcessing

class ZenithTokenizer:
    def __init__(self, model_filename="private_vocab_40k.json"):
        # 1. Path Resolution
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, model_filename)
        
        if not os.path.exists(model_path):
            model_path = model_filename 
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_filename}")
            
        # 2. Load Core Engine
        self.tokenizer = Tokenizer.from_file(model_path)
        
        # 3. FIX: THE TAB PRESERVATION SEQUENCE
        # We use a Sequence to ensure the tab (\t) is handled as a unique unit 
        # BEFORE the ByteLevel engine gets a chance to skip it.
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            # Split on tabs and keep them (isolated behavior)
            pre_tokenizers.Split(pattern="\t", behavior="isolated"),
            # Then apply ByteLevel to the chunks, ensuring no ghost space
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        ])
        
        # 4. FIX: Professional Decoder
        self.tokenizer.decoder = decoders.ByteLevel(
            add_prefix_space=False, 
            trim_offsets=False
        )
        
        # 5. Post-Processor (Standard SLM format)
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", 0), ("</s>", 1)],
        )
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # 6. WATERMARK (Stable 32-bit ID)
        self.signature_id = 271227292
        self.signature_text = "SKL_ZENITH_PROPRIETARY_2026"

    def encode(self, text):
        if not text: return []
        # add_special_tokens=False is safer for internal round-trip tests
        return self.tokenizer.encode(str(text), add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=True):
        # 1. Primary Decode
        decoded = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        # 2. THE STEM RECOVERY MAP
        manual_fixes = {
            "âĦı": "ℏ", "âĪĤ": "∂", "âĪĩ": "∇", "Î¨": "Ψ", "Î¦": "Φ", "âĪ®": "∮", 
            "âīĪ": "≈", "ÃĹ": "×", "âģ»": "⁻", "âĤĢ": "₀", "ÏĢ": "π", "âĪĢ": "∀", 
            "âĪĪ": "∈", "âĦĿ": "ℝ", "âĪĥ": "∃", "âī¡": "≡", "âĪŀ": "∞", "âĨĴ": "→",
            "Â²": "²", "Â³": "³", "âĪĨ": "∆", "âĪ´": "∝", "âĪ±": "±", "âĪ∓": "∓",
            "âīł": "≠", "âīħ": "≅", "âī¤": "≤", "âī¥": "≥", "âīŀ": "≪", "âīģ": "≫",
            "âĪ┤": "∴", "âĪµ": "∵", "âĪĦ": "∄", "âĪ¬": "¬", "âĪ§": "∧", "âĪ¨": "∨",
            "âĬķ": "⊕", "âĬĹ": "⊗", "âĬĻ": "⊙", "âĬĺ": "⊘", "âĬĽ": "⊛", "âĬŀ": "⊞",
            "âĬŁ": "⊟"
        }

        for mojibake, symbol in manual_fixes.items():
            decoded = decoded.replace(mojibake, symbol)

        return decoded

    def verify_authenticity(self):
        try:
            return self.signature_text in self.decode([self.signature_id])
        except:
            return False
