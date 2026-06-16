# ═══════════════════════════════════════════════════════════════
#  AI Security System — Dashboard Dockerfile
#  Multi-stage: build React app, serve with nginx
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: Build the React/Vite app ────────────────────────
FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies first (cached layer)
COPY package.json package-lock.json* ./
RUN npm ci --production=false

# Copy source and build
COPY . .
RUN npm run build

# ── Stage 2: Serve with nginx ────────────────────────────────
FROM nginx:alpine AS production

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our nginx config that proxies /api → backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built dashboard
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
