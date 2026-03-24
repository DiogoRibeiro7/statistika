import { defineConfig } from 'vitepress';

export default defineConfig({
  title: 'node_stats',
  description: 'Comprehensive statistical library for Node.js',
  base: '/node_stats/',
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Tutorials', link: '/tutorials/distributions' },
      { text: 'API Reference', link: '/api/' },
      { text: 'GitHub', link: 'https://github.com/DiogoRibeiro7/node_stats' },
    ],
    sidebar: {
      '/guide/': [
        { text: 'Getting Started', link: '/guide/getting-started' },
        { text: 'Installation', link: '/guide/installation' },
        { text: 'Quick Start', link: '/guide/quick-start' },
      ],
      '/tutorials/': [
        { text: 'Distributions', link: '/tutorials/distributions' },
        { text: 'Regression', link: '/tutorials/regression' },
        { text: 'Bayesian Inference', link: '/tutorials/bayesian' },
        { text: 'Time Series', link: '/tutorials/time-series' },
        { text: 'Survival Analysis', link: '/tutorials/survival' },
        { text: 'Causal Inference', link: '/tutorials/causal-inference' },
      ],
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/DiogoRibeiro7/node_stats' },
    ],
  },
});
