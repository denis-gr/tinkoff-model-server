FROM huggingface/transformers-pytorch-gpu

# 
WORKDIR /model

ADD . /model/
# 
COPY ./requirements.txt /model/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade --force-reinstall -r /model/requirements.txt

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]