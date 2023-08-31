# 
FROM python:3.9

# 
WORKDIR /model

ADD . /model/
# 
COPY ./requirements.txt /model/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /model/requirements.txt

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]