# Use official Python 3.11.11 slim base
FROM python:3.11.11-slim

WORKDIR /app

# Copy main code
COPY flagging_script.py .
COPY lexicon_functions.py .

# Copy lexicon files into image
COPY lexicons/ ./lexicons/

# Install dependencies
RUN pip install --no-cache-dir pandas numpy

# Default command
ENTRYPOINT ["python", "flagging_script.py"]
