import pandas
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig

prompt = """Generate a high-quality, engaging and professional social media post for a company called Itobuz in a descriptive format. Follow the example structure and ensure clarity, creativity, context awareness, and audience engagement.
    Context:
    - Platform: Instagram
    - Topic: Product launch of fittness band
    - Language: English
    - Word Limit: 250

    Requirements:
    - Craft a compelling opening that grabs attention.  
    - Highlight key details about the business or occasion.  
    - Maintain a consistent and engaging tone throughout.  
    - Use emojis for a more expressive post.
    - Tone should be determined by the platform.
    - Use persuasive language and storytelling where applicable.  
    - Include a strong call to action (CTA) to encourage engagement.  


    Now, generate a social media post using the provided context."""

base_model = AutoModelForCausalLM.from_pretrained('src/base_models/falcon1b/model')
tokenizer = AutoTokenizer.from_pretrained('src/base_models/falcon1b/tokenizer')

# Load LoRA adapters on top of the base model
model = PeftModel.from_pretrained(base_model, 'results/llm_results/fine_tuning_results_v1/checkpoint-115')

# Move to MPS (Mac) or CPU
# device = "mps"  # Use "cpu" if you don’t have Metal enabled
# model.to(device)

# Function for inference
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens = 500)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
# prompt = "Once upon a time"
result = generate_text(prompt)
print(result)