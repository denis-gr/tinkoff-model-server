from fastapi import FastAPI, Body

from model import LLM


app = FastAPI()

model = LLM()

@app.post("/complete")
async def read_item(prompt=Body()):
    return model.complete(prompt)
