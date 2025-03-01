FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p logs results experiments data

# Create a default config.ini with relative paths if one doesn't exist
RUN if [ ! -s config.ini ]; then \
    echo "[DATA]" > config.ini && \
    echo "x_data = data/Penguins_X.csv" >> config.ini && \
    echo "y_data = data/Penguins_y.csv" >> config.ini && \
    echo "" >> config.ini && \
    echo "[SPLIT_DATA]" >> config.ini && \
    echo "x_train = data/Train_Penguins_X.csv" >> config.ini && \
    echo "y_train = data/Train_Penguins_y.csv" >> config.ini && \
    echo "x_test = data/Test_Penguins_X.csv" >> config.ini && \
    echo "y_test = data/Test_Penguins_y.csv" >> config.ini && \
    echo "" >> config.ini && \
    echo "[RANDOM_FOREST]" >> config.ini && \
    echo "n_estimators = 100" >> config.ini && \
    echo "max_depth = None" >> config.ini && \
    echo "min_samples_split = 2" >> config.ini && \
    echo "min_samples_leaf = 1" >> config.ini && \
    echo "path = experiments/random_forest.sav" >> config.ini; \
    fi

EXPOSE 5000

CMD ["python", "src/api.py"]
