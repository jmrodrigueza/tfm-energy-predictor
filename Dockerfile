# Step 1: Build Angular
FROM node:20-slim AS angular-build
WORKDIR /app
COPY tfm-energy-predictor-web/package.json tfm-energy-predictor-web/package-lock.json ./
RUN npm install
COPY tfm-energy-predictor-web/ .
RUN npm run build --prod

# Step 2: Build Backend
FROM python:3.9-slim AS backend
WORKDIR /app
# Set your own API token from Hugging Face API at https://huggingface.co/
ENV API_TOKEN=_fill_your_own_API_token_here_

# Install Apache
RUN apt-get update && apt-get install -y apache2 libxml2-dev libapache2-mod-wsgi-py3
COPY httpd.conf /etc/apache2/sites-available/000-default.conf
RUN a2enmod proxy proxy_http rewrite

# Backend
WORKDIR /app/backend
COPY tfm-energy-predictor-backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY tfm-energy-predictor-backend/ .
# Frontend
COPY --from=angular-build /app/dist/tfm-energy-predictor-web/browser/ /var/www/html/
EXPOSE 5000
EXPOSE 8888

CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port 5000 > /var/log/uvicorn.log 2>&1 & apachectl -D FOREGROUND"]
