import { defineConfig } from "astro/config";

import sitemap from "@astrojs/sitemap";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { remarkReadingTime } from "./remark-reading-time.mjs";

// https://astro.build/config
export default defineConfig({
  site: "https://anyinlover.github.io/",
  integrations: [sitemap()],
  markdown: {
    remarkPlugins: [remarkMath, remarkReadingTime],
    rehypePlugins: [rehypeKatex],
  },
  i18n: {
    defaultLocale: "en",
    locales: ["en", "zh-cn"],
  },
});
