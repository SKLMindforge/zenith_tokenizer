import os
from tokenizers import Tokenizer, decoders, pre_tokenizers
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
        self.tokenizer.normalizer = None 
        
        # 1. Standard ByteLevel Setup
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True
        )
        
        self.tokenizer.decoder = decoders.ByteLevel(
            add_prefix_space=False, 
            trim_offsets=False
        )
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", 0), ("</s>", 1)],
        )
        self.vocab_size = self.tokenizer.get_vocab_size()

        # 2. INTERNAL PATCH: Tab Recovery Logic
        # We use a unique marker that is highly unlikely to appear in STEM text
        self.tab_placeholder = " [TAB_Z_MARKER] "
        # We search for an ID that corresponds to a single space to use as a bridge
        self.bridge_id = 221 # Standard Byte-Level Space 'Ġ'

    def encode(self, text):
        if not text: return []
        
        # FIX: Manual Tab Preservation
        # We replace actual tabs with a string the tokenizer CAN see
        # then encode it.
        processed_text = str(text).replace("\t", self.tab_placeholder)
        return self.tokenizer.encode(processed_text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=True):
        # 1. Standard Decode
        decoded = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        # 2. FIX: Convert Placeholders back to real Tabs
        decoded = decoded.replace(self.tab_placeholder, "\t")

        # 3. STEM RECOVERY MAP
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
            return "SKL_ZENITH_PROPRIETARY_2026" in self.decode([271227292])
        except:
            return False
