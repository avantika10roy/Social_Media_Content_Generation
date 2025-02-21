from fastapi import FastAPI
from src.prompts import llm_prompt
from pydantic import BaseModel


app = FastAPI()
class Prompt(BaseModel):
    company_name      : str
    platform          : str
    topic             : str
    extra_details     : str
    occasion          : str



@app.get("/")
def start():
    return {"Meaasge":"Welcome to LLMs"}

@app.post("/llm")
def gen_prompt(prompt : Prompt):
    company_name     = prompt.company_name
    platform         = prompt.platform
    topic            = prompt.topic
    extra_details    = prompt.extra_details
    occation         = prompt.occasion


    prompt = llm_prompt(platform        = platform,
                        topic           = topic,
                        company_name   = company_name,
                        extra_details  = extra_details,
                        occasion       = occation )
    return {"Generated Prompt ":prompt}
    


