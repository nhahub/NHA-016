FROM python:3.10-slim

WORKDIR /app

# تثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كود التطبيق والقوالب
COPY . .

# فتح البورت 5000
EXPOSE 5000

# تشغيل التطبيق
CMD ["python", "app.py"]