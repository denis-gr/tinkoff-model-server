from typing import Annotated

from fastapi import FastAPI, Body

from model import LLM


app = FastAPI()

model = LLM()

@app.post("/complete")
async def read_item(prompt: Annotated[str, Body()]):
    return model(prompt)
