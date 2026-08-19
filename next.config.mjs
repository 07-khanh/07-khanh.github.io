/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: { unoptimized: true },
  // basePath: '/your-repo-name', // Uncomment if deploying to a subpath repository like username.github.io/repo-name
};
export default nextConfig;
