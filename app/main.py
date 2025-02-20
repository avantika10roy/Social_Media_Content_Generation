from fastapi import FastAPI
from fastapi import HTTPException

app = FastAPI()

@app.get("/")
def opening():
    return {"message":"Welcome to Social Media Content Genaretor XD"}

@app.post("/llm")
def llm_op():
    return {"message":"LLM Started"}