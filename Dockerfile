FROM python:3.12-slim
LABEL authors="Dang"

WORKDIR /src

# Cài libomp + các công cụ biên dịch cần thiết cho wordcloud
RUN apt-get update && apt-get install -y \
    gcc \
    libomp-dev \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Copy model files
COPY W11/modelsComb/stage1_ocsvm_scaler.p /src/models/stage1_ocsvm_scaler.p
COPY W11/modelsComb/stage1_ocsvm_100k.p /src/models/stage1_ocsvm_100k.p
COPY W11/modelsComb/stage2_rf_scaler.p /src/models/stage2_rf_scaler.p
COPY W11/modelsComb/stage2_rf.p /src/models/stage2_rf.p
COPY server.py /src/server.py
COPY requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]
