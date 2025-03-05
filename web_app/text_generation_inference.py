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
        self.model = self.load_model(model_path)


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
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

        logging.info(f"Loading model from {model_path}...")
        model = Llama(model_path = model_path,
                      n_ctx=1024)

        return model
    
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
    - Encourage reactions, comments, or discussions.
    - The response **must** be at least 150 words long.
    - Avoid overly brief responses—explain the topic in detail.
    - Ensure the post contains **three structured paragraphs**:
        1. **Introduction** – Engaging hook.
        2. **Main Content** – Explanation of the brief.
        3. **Conclusion** – Call to action & hashtags.
    - Use emojis **sparingly and strategically**, ensuring platform relevance.
    - At the end, write exactly three hashtags relevant to the topic and industry.
    - End with a **strong Call to Action (CTA)**.
    - Learn from the example but not copy it.
    - The generated post should be ready to be posted online.

    ### Example Response:
    (The following example demonstrates an effective structure. Use it as a guideline for tone, flow, and hashtag usage, but do not copy it word-for-word.)
    🌟 Big news! 🚀 {company_name} is bringing something exciting your way.  
    Whether you're looking to {brief}, we've got exactly what you need.  
    Our latest update ensures {extra_details}, making it easier than ever for {target_audience} to benefit.  

    💡 Stay ahead of the curve and be part of the conversation!  
    Drop your thoughts in the comments and let us know what you think.  

    #StayAhead #YourSuccess #Innovation

    ### Response:
    """

        LLM_PROMPT = HEADING 
        return LLM_PROMPT
    def clean_hashtags(self,text, max_hashtags=5):

        hashtags = re.findall(r"#\w+", text)

        unique_hashtags = list(dict.fromkeys(hashtags))
    
        cleaned_text = re.sub(r"#\w+", "", text).strip()  
        cleaned_text += " " + " ".join(unique_hashtags[:max_hashtags])  
        return cleaned_text.strip()

    def generate_post(self, company_name:str,occasion: str, brief: str, topic:str, platform: str, target_audience: str, tone: str, extra_details: str = "", max_length: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = self.llm_prompt(company_name = company_name,
                                 occasion = occasion,
                                 brief = brief,
                                 topic = topic,
                                 platform = platform,
                                 target_audience = target_audience,
                                 tone = tone,
                                 extra_details = extra_details)

        with torch.no_grad():
            output = self.model(prompt,
                                max_tokens           = 2048, 
                                top_p                = 0.85, 
                                top_k                = 85, 
                                temperature          = 0.6,
                                repeat_penalty       = 1.1,
                                stop                 = ["###"])
        
        clean_text = self.clean_hashtags(text=output['choices'][0]['text'])

        return clean_text

# Initialize Model
MODEL_PATH = "../models/falcon_finetuned_quantized/falcon3_1b_instruct_finetuned_v11.gguf"
text_generator  = SocialMediaPostGenerator(MODEL_PATH, device='cpu')
