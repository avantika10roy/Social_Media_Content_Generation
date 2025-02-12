    

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

def llm_finetuning_prompt(platform:str, heading:str, content:str, tone:str, lang:str, word_lim:int) -> str :
    LLM_PROMPT = f"""Generate a high-quality, engaging and professional social media post for a {company_name} in a descriptive format. Follow the example structure and ensure clarity, creativity, context awareness, and audience engagement.

Context: 
- Post Heading: {heading}
- Platform: {platform}  
- Tone: Friendly
- Language: English  
- Word Limit: 250 words  

Response:
- Content:{content}
- Emoji:{emojis}
- Hashtags:{hashtags}

"""

    return LLM_PROMPT
# -------------------------------------

