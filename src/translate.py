from transformers import MarianMTModel, MarianTokenizer
import torch

class LocalTranslator:
    def __init__(self, source_lang="en", target_lang="tr"):
        # HATA DÜZELTMESİ:
        # Standart isim "opus-mt-en-tr" mevcut değil. 
        # İngilizce -> Türkçe için "opus-mt-tc-big-en-tr" kullanılıyor.
        if source_lang == "en" and target_lang == "tr":
            self.model_name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
        else:
            # Diğer diller için standart isimlendirmeyi dene (Örn: tr-en çalışıyor)
            self.model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            
        print(f"Model yükleniyor: {self.model_name} (CPU)...")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name, use_safetensors=True)
        except OSError as e:
            print(f"\nHATA: '{self.model_name}' modeli Hugging Face'de bulunamadı.")
            print("Lütfen dil kodlarını veya model ismini kontrol edin.")
            raise e
        
        self.device = torch.device("cpu")
        self.model.to(self.device)
        print("Model başarıyla yüklendi.")

    def translate(self, text_list):
        if isinstance(text_list, str):
            text_list = [text_list]

        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            translated_tokens = self.model.generate(**inputs)

        decoded_translations = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        return decoded_translations

if __name__ == "__main__":
    # Test
    translator = LocalTranslator(source_lang="en", target_lang="tr")
    
    sentences = [
        "Hello, this is a test.",
        "How is your day going?",
        "Transformers library makes translation easy!",
        "This is a sample sentence for translation.",
        "Let's see how well this model performs.",   # write a hard sentence for translation
        "The quick brown fox jumps over the lazy dog.",
        "Artificial Intelligence is transforming the world.",  
        "Can you translate this sentence into Turkish?",
        "What are the challenges in machine translation?",
        "Natural Language Processing is a fascinating field."
    ]
    
    results = translator.translate(sentences)
    for res in results:
        print(res)