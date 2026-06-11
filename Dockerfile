FROM python:3.10

WORKDIR /app

# System dependencies
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

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure streamlit is installed (important safety step)
RUN pip install streamlit

# Copy project
COPY . .

# Render expects dynamic port sometimes → safer setup
ENV PORT=7860

EXPOSE 7860

# Start app
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0
