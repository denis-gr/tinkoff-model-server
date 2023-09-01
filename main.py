from fastapi import FastAPI, Body

from model import LLM


app = FastAPI()

model = LLM()

@app.post("/complete")
async def read_item(prompt: str = Body()):
    return { "text": model.complete(prompt) }

