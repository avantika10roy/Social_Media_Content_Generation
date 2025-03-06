# Dependencies
import re
import os
import torch
import logging
from peft import PeftModel
from transformers import AutoTokenizer
from llama_cpp import Llama
from transformers import AutoModelForCausalLM


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SocialMediaPostGenerator:
    def __init__(self, model_path: str, device: str = "auto"):
        self.device                = self._set_device(device)
        self.model, self.tokenizer = self.load_model(model_path)


    def _set_device(self, device: str) -> str:
        if (device == "auto"):
            if torch.cuda.is_available():
                return "cuda"
            
            elif torch.backends.mps.is_available():
                return "mps"
            
            else:
                return "cpu"
        
        return device


    def load_model(self, model_path: str):
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"Model path '{model_path}' does not exist.")
        if (not os.path.isdir(model_path)):
            base_model             = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path ='tiiuae/Falcon3-1B-Instruct',
                                                                          torch_dtype                   = torch.float16 if self.device == 'cuda' else torch.float32)
            tokenizer              = AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Instruct')

            tokenizer.pad_token_id = tokenizer.eos_token_id

            base_model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

        else:
            logging.info(f"Loading tokenizer from {model_path}...")
            tokenizer              = AutoTokenizer.from_pretrained(model_path)

            logging.info(f"Loading model from {model_path}...")
            base_model             = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_path,
                                                                          torch_dtype                   = torch.float16 if self.device == 'cuda' else torch.float32)
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if (not hasattr(base_model, "peft_config")):
            logging.info("Applying LoRA adapter...")
            model                  = PeftModel.from_pretrained(base_model, 'models/llm/adapter_2')
        
        else:
            model                  = base_model
            logging.info("LoRA adapter already applied.")

        model.to(self.device)
        model.eval()

        return model, tokenizer
    
    def llm_prompt(self,platform:str, brief:str, company_name:str, target_audience:str="Followers", 
                extra_details:str="", occasion:str=None, topic:str=None, tone:str="Formal", 
                lang:str='English', word_lim:int=250) -> str:
        

        HEADING = f"""
    ### Task:
    Generate a social media post for a company called "{company_name}" in a **descriptive format** in at least one hundred words.
    The content must align with the brand's identity, target audience, and focus on the **brief**.
        
    ### Context:
    - Platform: {platform}
    - Topic: **{occasion if topic == "" else topic}**
    - Brief: {brief}
    - Extra Details: {extra_details}
    - Language: {lang}
    - Word Limit: {word_lim}
    - Tone: {tone}
    - Target Audience: {target_audience}

    ### Requirements:
    - The first sentence must grab attention.
    - The tone, vocabulary, and style must reflect the brand’s voice.
    - Clearly convey insights about **brief** in a structured manner.
    - Don't add any irrelevant information by your own.
    - Don't hallucinate during the post generation.
    - Don't change {company_name} during the post generation.
    - Encourage reactions, comments, or discussions.
    - Don't give summary. Just the give main content.
    - Complete the post within {word_lim} words.
    - Only respond with information that is well-documented and verifiable. If the answer is uncertain or unknown, state that explicitly.
    - Avoid overly brief responses—explain the topic in moderate length.
    - Ensure the post contains **three structured paragraphs**:
        1. **Introduction** – Engaging hook.
        2. **Main Content** – Explanation of the brief.
        3. **Conclusion** – Call to action & hashtags.
    - Avoid trailing or incomplete phrases.
    - Use emojis **sparingly and strategically**, ensuring platform relevance.
    - At the end, write exactly three hashtags relevant to the topic and brief.
    - End with a **strong Call to Action (CTA)**.
    - Learn from the example but not copy it.
    - The generated post should be ready to be posted online.

    ### Response:
    """
    ### Example Response:
    # (The following example demonstrates an effective structure. Use it as a guideline for tone, flow, and hashtag usage, but do not copy it word-for-word.)
    # 🌟 Big news! 🚀 {company_name} is bringing something exciting your way.  
    # Whether you're looking to {brief}, we've got exactly what you need.  
    # Our latest update ensures {extra_details}, making it easier than ever for {target_audience} to benefit.  

    # 💡 Stay ahead of the curve and be part of the conversation!  
    # Drop your thoughts in the comments and let us know what you think.  

    # #StayAhead #YourSuccess #Innovation

        LLM_PROMPT = HEADING 
        return LLM_PROMPT
    def clean_hashtags(self,text, max_hashtags=5):

        hashtags = re.findall(r"#\w+", text)

        unique_hashtags = list(dict.fromkeys(hashtags))
    
        cleaned_text = re.sub(r"#\w+", "", text).strip()  
        cleaned_text += " " + " ".join(unique_hashtags[:max_hashtags])  
        return cleaned_text.strip()
    
    def extract_response(self, text):
        match = re.search(r"### Response:\n(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def replace_continuous_dots(self, text):
        # Remove <assistant> from the text
        text = re.sub(r'<\|assistant\|>', '', text)

        # Remove markdown formatting (like triple backticks)
        text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)  # Remove code blocks
        text = re.sub(r'\*{1,2}(\S.*?)\*{1,2}', r'\1', text)  # Remove italic and bold markdown (**text**, *text*)
        text = re.sub(r'_([^\s][^_]*)_', r'\1', text)  # Remove underscores for italics (_text_)
        text = re.sub(r'~{2}(.*?)~{2}', r'\1', text)  # Remove strikethrough (~~text~~)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links [text](url)
        text = re.sub(r'\n?#{2,}\s+', '', text)  # Remove headings (## Heading)
        text = re.sub(r'(?<=\s)#(?!\S)', '', text)  # Remove standalone hashtags
        
        # Remove sequences of hashtags (e.g., ###, ##, ####, etc.)
        text = re.sub(r'(#+)\1+', '', text)
        
        # Remove single hashtags not followed by text
        text = re.sub(r'#(?=\s|$|[^\w])', '', text)
        
        # Remove continuous delimiters (like ..., ---, etc.) but keep single dots and commas after text
        text = re.sub(r'([*\-_=+|])\1*', '', text)  # Remove continuous occurrences of *, -, _, =, +, |
        text = re.sub(r'(?<!\w)[.,]+(?!\w)', '', text)  # Remove continuous dots and commas not after a word
        # text = re.sub(r'(?<=\w)[.,](?=\s|$)', '', text)  # Keep single dots and commas after words

        return text

    def generate_post(self, company_name:str,occasion: str, brief: str, topic:str, platform: str, target_audience: str, tone: str, extra_details: str = "", max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = self.llm_prompt(company_name = company_name,
                                 occasion = occasion,
                                 brief = brief,
                                 topic = topic,
                                 platform = platform,
                                 target_audience = target_audience,
                                 tone = tone,
                                 extra_details = extra_details)
        
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs,
                                         min_new_tokens       = 100,
                                         max_new_tokens       = 1000, 
                                         do_sample            = True, 
                                         top_p                = 0.95, 
                                         top_k                = 40, 
                                         temperature          = 0.3, 
                                         no_repeat_ngram_size = 3, 
                                         repetition_penalty   = 1.2,
                                         pad_token_id         = self.tokenizer.eos_token_id,
                                         eos_token_id         = self.tokenizer.eos_token_id)
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        clean_text = self.extract_response(generated_text)

        clean_text = self.clean_hashtags(clean_text)
        clean_text = clean_text.replace('# ', '').replace('**', '').replace('•', '').replace('  ', '')
        print(clean_text)
        clean_text = self.replace_continuous_dots(clean_text)
        return clean_text

# Initialize Model
MODEL_PATH = "./models/llm/base_model"

text_generator  = SocialMediaPostGenerator(MODEL_PATH, device='cpu')
