def llm_prompt(platform: str, topic: str, company_name: str, lang: str = 'English', word_lim: int = 250) -> str:
    prompt = f'''
    ## CONTEXT ##
    Generate a high-quality, engaging, and professional social media post for {company_name}. Ensure clarity, creativity, context awareness, and audience engagement.
    
    - Platform: {platform}  
    - Topic: {topic}
    - Language: {lang}  
    - Word Limit: {word_lim}
    
    ## TASK ##
    Craft a compelling social media post using the provided context.
    
    ## GUIDELINES ##
    1. Start with an attention-grabbing opening.
    2. Highlight key details about the business or occasion.
    3. Maintain a consistent and engaging tone suitable for {platform}.
    4. Use persuasive language and storytelling where applicable.
    5. Include a strong call to action (CTA) to encourage engagement.
    6. Ensure the post is structured clearly and professionally.
    
    ## OUTPUT FORMAT ##
    Respond with a structured social media post following these guidelines.
    '''
    return prompt


def llm_finetuning_prep(data_dict: dict) -> str:
    company_name = 'Itobuz'
    platform = data_dict.get('platform', 'Unknown Platform')
    heading = data_dict.get('post_heading', 'No Heading Provided')
    content = data_dict.get('post_content', 'No Content Provided')
    hashtags = ", ".join(data_dict.get('hashtags', [])) or "No Hashtags"
    emojis = ", ".join(data_dict.get('emoji', [])) or "No Emoji Needed"
    raw_post_content = data_dict.get('raw_content', 'No Raw Content Provided')

    prompt = f'''
    ## CONTEXT ##
    Prepare data for fine-tuning by structuring a social media post dataset.
    
    - Company Name: {company_name}
    - Platform: {platform}
    - Heading: {heading}
    - Content: {content}
    - Emojis: {emojis}
    - Hashtags: {hashtags}
    
    ## TASK ##
    Format the data for fine-tuning by ensuring consistency in structure and clarity.
    
    ## OUTPUT FORMAT ##
    - Response: {raw_post_content}
    '''
    return prompt
