FROM python:3.7-slim

RUN mkdir /frontend
WORKDIR /frontend
RUN mkdir /data
RUN mkdir /models

COPY frontend/ /frontend
COPY ../data ./data
COPY ../models ./models
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl --no-cache-dir
# RUN pip install tensorflow==2.2.0
RUN  rm -r tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl
RUN rm -rf data/test/*

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py"]
