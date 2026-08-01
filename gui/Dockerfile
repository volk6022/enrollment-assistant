# syntax=docker/dockerfile:1

# --- build stage -------------------------------------------------------
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# --- runtime stage -------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

# Defaults so `docker run` without extra env still boots; docker-compose.yml
# (T-11) is expected to override BACKEND_HOST/BACKEND_PORT to point at the
# backend service, and GUI_PORT to whatever is set in .env (FR-32).
ENV GUI_PORT=8080 \
    BACKEND_HOST=backend \
    BACKEND_PORT=8000

COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

# nginx:alpine's entrypoint renders /etc/nginx/templates/*.template into
# /etc/nginx/conf.d/*.conf via envsubst before starting nginx — no custom
# entrypoint script needed.
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:${GUI_PORT}/ >/dev/null 2>&1 || exit 1
