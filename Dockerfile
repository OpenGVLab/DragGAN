FROM python:3.7

WORKDIR /app
COPY . .
EXPOSE 7860

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python", "-m", "draggan.web", "--ip", "0.0.0.0"]
