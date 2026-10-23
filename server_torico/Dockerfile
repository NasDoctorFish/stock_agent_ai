FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=optional

COPY . .

RUN npm run build

ENV PORT=4000
EXPOSE 4000

CMD ["node", "dist/server.js"]
