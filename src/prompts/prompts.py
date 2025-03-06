    

# -------- Done By Manu --------------
def llm_prompt(platform:str, brief:str, company_name:str, target_audience:str="Followers", extra_details:str="", occasion:str=None, topic:str=None,tone:str="Formal", lang:str = 'English', word_lim:int = 250) -> str :
    HEADING = f"""
    ### Generate a high-quality, engaging and **platform-optimized** social media post for a company called "{company_name}" in a descriptive format.
    ### The company name **must remain exactly as given** and should never be changed or abbreviated.
    ### The post should, align with the company's audience, fulfil the requirements and ensure clarity, creativity, **context awareness**, and audience engagement."""
    
    if topic is None:
        CONTEXT = f"""
    ### Context:
    - Platform: {platform}
    - Occasion: **{occasion}**
    - Brief: {brief}
    - Extra Details: {extra_details}
    - Language: {lang}
    - Word Limit: {word_lim}
    - Tone: {tone}
    - Target Audience: {target_audience}"""
    else:
        CONTEXT = f"""
    ### Context:
    - Platform: {platform}
    - Topic: **{topic}**
    - Brief: {brief}
    - Extra Details: {extra_details}
    - Language: {lang}
    - Word Limit: {word_lim}
    - Tone: {tone}
    - Target Audience: {target_audience}"""

    REQUIREMENTS = """### Writing Requirements:
    - Opening should be compelling and must grab attention.  
    - Key details about the business or occasion must be highlighted.  
    - A consistent and engaging tone throughout MUST be maintained.  
    - Emoji usage MUST be dependent on the platform.
    - **Any hashtag provided in the "Extra Details" MUST be used.**  
    - **If no hashtag is provided in "Extra Details", at most 5 hashtags MUST be generated based on the topic.**  
    - Persuasive language and storytelling should be used where applicable.  
    - A strong, platform-appropriate call to action (CTA) should be included.  
    - Response MUST be complete, well-structured, and does not cut off mid-sentence.
    - Generated post should be ready to be posted online without placeholders or further edits.

    ### Response:
    - // Your response will go here. Replace it with your crafted content.
    """

    print(CONTEXT)

    LLM_PROMPT = HEADING + CONTEXT + REQUIREMENTS

    return LLM_PROMPT

def llm_finetuning_prep(data_dict: dict) -> str :

    # parsing dictionary for formatting
    company_name = 'Itobuz'
    platform = data_dict.get('platform')
    heading = data_dict.get('post_heading')
    content = data_dict.get('post_contents')
    hashtags = data_dict.get('hashtags')
    emojis = data_dict.get('emoji')
    emojis = ", ".join(emojis) if len(data_dict.get('emoji'))>0 else "No emoji needed"
    hashtags = ", ".join(hashtags)
    raw_post_content = data_dict.get('raw_post_content')

    context = f"""-Platform:{platform}\n-Topic:{heading}\n-Language:English"""

    question = f"""Generate a social media post for a company called "{company_name}". """

    prompt_template = f"""<|user|>\nContext:\n{context}\n\n{question}<|assistant|>\n{raw_post_content}<|endoftext|>"""


    return prompt_template
# -------------------------------------