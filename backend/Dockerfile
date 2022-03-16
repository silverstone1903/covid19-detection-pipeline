FROM python:3.7-slim

RUN mkdir /backend

COPY ./backend/requirements.txt /backend/requirements.txt 
RUN pip install -r backend/requirements.txt --no-cache-dir


COPY ./backend /backend
WORKDIR /backend
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]