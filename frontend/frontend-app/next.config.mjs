import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname);

const nextConfig = {
  webpack(config) {
    config.resolve.alias["@"] = projectRoot;
    return config;
  },
  turbopack: {
    resolveAlias: {
      "@": projectRoot,
    },
  },
};

export default nextConfig;
