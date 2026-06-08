FROM python:3.10

WORKDIR /app

# System dependencies for dlib + opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Hugging Face port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py",
     "--server.port=7860",
     "--server.address=0.0.0.0"]