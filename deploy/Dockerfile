# Live demo: FastAPI + web page for the credit-risk engine (Hugging Face Spaces, Docker SDK)
FROM python:3.14-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/code

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "deploy.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
