/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'server', 
        port: '8080',
        pathname: '/static/**',
      },
    ],
  },
};

module.exports = nextConfig;
