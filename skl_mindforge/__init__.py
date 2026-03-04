import os
from tokenizers import Tokenizer, decoders
from tokenizers.processors import TemplateProcessing

class ZenithTokenizer:
    def __init__(self, model_filename="private_vocab_40k.json"):
        # 1. Robust Path Resolution
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, model_filename)
        
        if not os.path.exists(model_path):
            model_path = model_filename 
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_filename}")
            
        # 2. Load Core Engine
        self.tokenizer = Tokenizer.from_file(model_path)
        
        # 3. FIX: Stop Deleting Whitespace
        # We set add_prefix_space=False and use ByteLevel to preserve EXACT formatting
        self.tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=False)
        
        # 4. Standard SLM Post-Processing
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", 0), ("</s>", 1)], # Corrected ID mapping from your JSON
        )
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # 5. WATERMARK (Updated to the stable 32-bit ID)
        self.signature_id = 271227292
        self.signature_text = "SKL_ZENITH_PROPRIETARY_2026"

    def encode(self, text):
        # We ensure text is string to prevent crashes
        return self.tokenizer.encode(str(text)).ids

    def verify_authenticity(self):
        try:
            # Using decode to ensure the watermark handles any byte-encoding issues
            return self.signature_text in self.decode([self.signature_id])
        except:
            return False

    def decode(self, ids, skip_special_tokens=True):
        # 1. Primary Decode (Preserves the raw bytes)
        decoded = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        # 2. THE STEM RECOVERY MAP (Your Math/Science Armor)
        # This fixes "Mojibake" where UTF-8 bytes are misread
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

        # 3. FIX: No more aggressive stripping!
        # We removed .strip() and the "clean" logic that deleted necessary spaces/commas
        return decoded

zenith_tokenizer = ZenithTokenizer()
