# Use an official Node.js runtime as a parent image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy all source files
COPY . .

# Expose Next.js default dev port
EXPOSE 3000

# Run Next.js in development mode with hot‑reload
CMD ["npm", "run", "dev"]
