FROM node:20-alpine
WORKDIR /srv/frontend
COPY package.json ./
EXPOSE 5173
