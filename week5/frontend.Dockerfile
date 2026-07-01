# --- Build stage: compile the React app ---
FROM node:20-alpine AS build
WORKDIR /app
COPY chat-ui/package*.json ./
RUN npm install
COPY chat-ui/ .
RUN npm run build

# --- Serve stage: nginx serving static files ---
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY chat-ui/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
