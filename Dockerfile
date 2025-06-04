FROM python:3.10

WORKDIR /app

RUN python -m venv /env
ENV PATH="/env/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


COPY . .
EXPOSE 8000

CMD ["uvicorn", "verify:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]