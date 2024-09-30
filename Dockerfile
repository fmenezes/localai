FROM python:3.12
WORKDIR /opt/app
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD app.py .

CMD ["python", "app.py"]