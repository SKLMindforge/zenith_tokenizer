import os
from tokenizers import Tokenizer, decoders
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
        
        # Enable the native ByteLevel decoder for base recovery
        self.tokenizer.decoder = decoders.ByteLevel()
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s> $B </s>",
            special_tokens=[("<s>", 1), ("</s>", 2)],
        )
        self.vocab_size = self.tokenizer.get_vocab_size()

    def decode(self, ids, skip_special_tokens=True):
        """
        VERSION 0.1.7: The Encyclopedia Decoder.
        Manually maps and recovers a massive library of scientific symbols.
        """
        # 1. Primary Decode using native ByteLevel
        decoded = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

        # 2. THE MANUAL RECOVERY MAP
        # This fixes the 'mojibake' for the specific symbols you provided.
        # Note: Native ByteLevel handles many, but this acts as the 'Steel Guard'.
        manual_fixes = {
            "âĦı": "ℏ", "âĪĤ": "∂", "âĪĩ": "∇", "Î¨": "Ψ", "Î¦": "Φ", "âĪ®": "∮", 
            "âīĪ": "≈", "ÃĹ": "×", "âģ»": "⁻", "âĤĢ": "₀", "ÏĢ": "π", "âĪĢ": "∀", 
            "âĪĪ": "∈", "âĦĿ": "ℝ", "âĪĥ": "∃", "âī¡": "≡", "âĪŀ": "∞", "âĨĴ": "→",
            "Â²": "²", "Â³": "³", "âĪĨ": "∆", "âĪ´": "∝", "âĪ±": "±", "âĪ∓": "∓",
            "âīł": "≠", "âīħ": "≅", "âī¤": "≤", "âī¥": "≥", "âīŀ": "≪", "âīģ": "≫",
            "âĪ┤": "∴", "âĪµ": "∵", "âĪĦ": "∄", "âĪ¬": "¬", "âĪ§": "∧", "âĪ¨": "∨",
            "âĬķ": "⊕", "âĬĹ": "⊗", "âĬĻ": "⊙", "âĬĺ": "⊘", "âĬĽ": "⊛", "âĬŀ": "⊞",
            "âĬŁ": "⊟", "âĪħ": "∅", "âĪī": "∉", "âĪĭ": "∋", "âĪĮ": "∌", "âĪĤ": "⊂",
            "âĪĽ": "⊃", "âĪĨ": "⊆", "âĪ©": "⊇", "âĬĦ": "⊄", "âĬħ": "⊅", "âĪª": "∪",
            "âĪ«": "∩", "âĪĸ": "∖", "âĪ«": "∬", "âĪ¬": "∭", "âĪ¯": "∯", "âĪ°": "∰",
            "âĪĳ": "∑", "âĪı": "∏", "âĪĲ": "∐", "âĪļ": "√", "âĪĻ": "∛", "âĪĽ": "∜",
            "âĪŁ": "∟", "âĪ¤": "∠", "âĪ¥": "∡", "âĪ¦": "∢", "âĪ¥": "∥", "âĪ¦": "∦",
            "âĪ¥": "⊥", "âĬĶ": "⊢", "âĬķ": "⊣", "âĬ¤": "⊤", "âĬ§": "⊨", "âĬ¨": "⊩",
            "âĬ©": "⊪", "âĬª": "⊫", "âĬ«": "⊬", "âĬ¬": "⊭", "âĬ®": "⊮", "âĬ¯": "⊯"
        }

        # Apply the map
        for mojibake, symbol in manual_fixes.items():
            decoded = decoded.replace(mojibake, symbol)

        # 3. Final Punctuation and Space Polish
        clean = decoded.replace(' ,', ',').replace(' .', '.').replace(' - ', '-')
        
        return clean.strip()

zenith_tokenizer = ZenithTokenizer
