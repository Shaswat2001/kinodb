import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://Shaswat2001.github.io',
  base: '/kinodb',
  integrations: [
    starlight({
      title: 'kinodb',
      description: 'A high-performance trajectory database for robot learning.',
      favicon: '/favicon.svg',
      logo: {
        light: './public/logo-light.svg',
        dark: './public/logo-dark.svg',
        replacesTitle: false,
      },
      customCss: ['./src/styles/custom.css'],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/Shaswat2001/kinodb',
        },
      ],
      editLink: {
        baseUrl: 'https://github.com/Shaswat2001/kinodb/edit/main/kinodb-docs/',
      },
      sidebar: [
        {
          label: 'Start Here',
          items: [
            { label: 'The Problem', slug: '' },
            { label: 'Why kinodb?', slug: 'guides/why-kinodb' },
            { label: 'Installation', slug: 'guides/installation' },
            { label: 'Quick Start', slug: 'guides/quickstart' },
          ],
        },
        {
          label: 'Build A Dataset',
          items: [
            { label: 'Ingesting Data', slug: 'guides/ingestion' },
            { label: 'KQL Queries', slug: 'guides/kql' },
            { label: 'Dataset Mixtures', slug: 'guides/mixtures' },
          ],
        },
        {
          label: 'Train And Serve',
          items: [
            { label: 'PyTorch Training', slug: 'guides/pytorch' },
            { label: 'Remote Serving', slug: 'guides/remote' },
          ],
        },
        {
          label: 'Results',
          items: [
            { label: 'IO Performance', slug: 'benchmarks/io' },
            { label: 'Training Pipeline', slug: 'benchmarks/training' },
            { label: 'Correctness', slug: 'benchmarks/correctness' },
          ],
        },
        {
          label: 'Technical Reference',
          items: [
            { label: 'CLI Commands', slug: 'reference/cli' },
            { label: 'Python API', slug: 'reference/python-api' },
            { label: 'File Format', slug: 'reference/file-format' },
            { label: 'Architecture', slug: 'reference/architecture' },
          ],
        },
      ],
    }),
  ],
});
