# Live demo of the credit-risk engine (FastAPI + web page). Used by Render (render.yaml);
# also works on any Docker host: docker build -t credit-risk . && docker run -p 7860:7860 credit-risk
FROM python:3.14-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /home/user/code

COPY --chown=user deploy/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# only what the engine and the app need (no data)
COPY --chown=user src/__init__.py src/__init__.py
COPY --chown=user src/engine src/engine
COPY --chown=user src/modeling/__init__.py src/modeling/decision.py src/modeling/reason_codes.py src/modeling/
COPY --chown=user src/preprocessing_v2/__init__.py src/preprocessing_v2/config.py src/preprocessing_v2/features.py src/preprocessing_v2/
COPY --chown=user models/final/04_scoring_package/bundle models/final/04_scoring_package/bundle
COPY --chown=user deploy/__init__.py deploy/__init__.py
COPY --chown=user deploy/app deploy/app

EXPOSE 7860
CMD ["sh", "-c", "uvicorn deploy.app.main:app --host 0.0.0.0 --port ${PORT}"]
