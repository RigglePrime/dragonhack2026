FROM python:3.14-slim

WORKDIR /app
COPY ./main.py .
COPY ./requirements.txt .
COPY ./eud_cp_slop/eudem_slop_3035_europe.tif ./eud_cp_slop/eudem_slop_3035_europe.tif
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT [ "python", "main.py" ]
