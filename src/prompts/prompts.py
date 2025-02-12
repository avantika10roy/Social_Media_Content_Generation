    

# -------- Done By Manu --------------
def llm_prompt(platform:str, theme:str, target:str, tone:str, lang:str, word_lim:int) -> str :
    LLM_PROMPT = f"""Generate a high-quality, engaging social media post for a business in a descriptive format. Follow the example structure and ensure clarity, creativity, and audience engagement.

    Context:
    - Platform: {platform}  
    - Theme: {theme}
    - Target Audience: {target}  
    - Tone: {tone}
    - Language: {lang}  
    - Word Limit: {word_lim}

    Requirements:
    - Craft a compelling opening that grabs attention.  
    - Highlight key details about the business or occasion.  
    - Maintain a consistent and engaging tone throughout.  
    - Use persuasive language and storytelling where applicable.  
    - Include a strong call to action (CTA) to encourage engagement.  


    Now, generate a social media post using the provided context."""

    return LLM_PROMPT

def llm_finetuning_prep(data_dict: dict) -> str :
    
    # parsing dictionary for formatting
    company_name = 'Itobuz'
    platform = data_dict.get('platform')
    heading = data_dict.get('post_heading')
    content = data_dict.get('post_content')
    hashtags = data_dict.get('hashtags')
    emojis = data_dict.get('emoji')
    emojis = ", ".join(emojis)
    hashtags = ", ".join(hashtags)
    raw_post_content = data_dict.get('raw_content')
    
    
    # genrating data format from the finetuning
    DATA_FORMAT = f""" 
    - Company Name : {company_name}
    - Platform: {platform}
    - Heading : {heading} 
    - Content : {content}
    - emojis : {emojis}
    - hashtags : {hashtags}
    - Response: {raw_post_content}"""

    return DATA_FORMAT
# -------------------------------------