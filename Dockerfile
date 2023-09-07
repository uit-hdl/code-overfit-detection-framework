FROM python:3.10-slim-bookworm

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements_train.txt requirements.txt
RUN pip install -r requirements.txt

# Run the application:
COPY condssl condssl
COPY network network
COPY "train_model" "train_model"
COPY run_train.sh run_train.sh

ENTRYPOINT ["./run_train.sh"]
#CMD ["python", "train_model/train_ssl.py", "--out-dir", "/output", "--workers", "16", "--batch-size", "32", "--batch_slide_num", "4", "--data-dir", "/Data/lung_scc/", "--condition", "True"]
#ipython train_model/train_ssl.py -- --out-dir ./out --workers 16 --batch-size 64 --batch_slide_num 4 --data-dir ~/TCGA_LUSC/preprocessed/by_class/lung_scc/ --condition False
