import { defineConfig } from "vitest/config";

export default defineConfig({
  test: { environment: "jsdom", include: ["lib/**/*.test.ts", "lib/**/*.test.tsx"] },
  resolve: { alias: { "@": new URL("./", import.meta.url).pathname } },
});
