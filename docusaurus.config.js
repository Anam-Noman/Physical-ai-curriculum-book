// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to parse and validate the configuration
// This is used by Docusaurus to configure and deploy the website

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Curriculum',
  tagline:
    'Bridging digital AI models with physical robotic bodies through simulation, perception, planning, and action',
  favicon: 'img/favicon.ico',

  // Production URL
  url: 'https://anam-noman.github.io',

  // Base URL for Vercel deployment
  baseUrl: '/',

  // GitHub Pages deployment config
  organizationName: 'Anam-Noman',
  projectName: 'Physical-ai-curriculum-book',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/Anam-Noman/Physical-ai-curriculum-book/edit/main/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotic Book',
        logo: {
          alt: 'AI Icon',
          src: 'https://static.vecteezy.com/system/resources/previews/027/396/758/non_2x/modern-technology-cybersecurity-icon-crop-out-free-png.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'curriculumSidebar',
            position: 'left',
            label: 'Modules',
          },
          {
            href: 'https://github.com/Anam-Noman?tab=repositories',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Anam-Noman?tab=repositories',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Curriculum. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),
};

module.exports = config;




// // @ts-check
// // `@type` JSDoc annotations allow IDEs and type checkers to parse and validate the configuration
// // This is used by Docusaurus to configure the website

// /** @type {import('@docusaurus/types').Config} */
// const config = {
//   title: 'Physical AI & Humanoid Robotics Curriculum',
//   tagline:
//     'Bridging digital AI models with physical robotic bodies through simulation, perception, planning, and action',
//   favicon: 'img/favicon.ico',

//   // Production URL
//   url: 'https://anam-noman.github.io',

//   // Base URL (MUST match repo name, case-sensitive)
//   baseUrl: '/Physical-ai-curriculum-book/',

//   // GitHub Pages deployment config
//   organizationName: 'Anam-Noman',
//   projectName: 'Physical-ai-curriculum-book',

//   onBrokenLinks: 'throw',
//   onBrokenMarkdownLinks: 'warn',

//   i18n: {
//     defaultLocale: 'en',
//     locales: ['en'],
//   },

//   presets: [
//     [
//       'classic',
//       /** @type {import('@docusaurus/preset-classic').Options} */
//       ({
//         docs: {
//           sidebarPath: require.resolve('./sidebars.js'),
//           editUrl:
//             'https://github.com/Anam-Noman/Physical-ai-curriculum-book/edit/main/',
//         },
//         blog: false,
//         theme: {
//           customCss: require.resolve('./src/css/custom.css'),
//         },
//       }),
//     ],
//   ],

//   themeConfig:
//     /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
//     ({
//       image: 'img/docusaurus-social-card.jpg',
//       navbar: {
//         title: 'Physical AI & Humanoid Robotic Book',
//         logo: {
//           alt: 'AI Icon',
//           src: 'https://static.vecteezy.com/system/resources/previews/027/396/758/non_2x/modern-technology-cybersecurity-icon-crop-out-free-png.png',
//         },
//         items: [
//           {
//             type: 'docSidebar',
//             sidebarId: 'curriculumSidebar',
//             position: 'left',
//             label: 'Modules',
//           },
//           {
//             href: 'https://github.com/Anam-Noman?tab=repositories',
//             label: 'GitHub',
//             position: 'right',
//           },
//         ],
//       },
//       footer: {
//         style: 'dark',
//         links: [
//           {
//             title: 'Modules',
//             items: [
//               {
//                 label: 'Introduction',
//                 to: '/docs/intro',
//               },
//             ],
//           },
//           {
//             title: 'Community',
//             items: [
//               {
//                 label: 'Stack Overflow',
//                 href: 'https://stackoverflow.com/questions/tagged/docusaurus',
//               },
//               {
//                 label: 'Discord',
//                 href: 'https://discordapp.com/invite/docusaurus',
//               },
//               {
//                 label: 'Twitter',
//                 href: 'https://twitter.com/docusaurus',
//               },
//             ],
//           },
//           {
//             title: 'More',
//             items: [
//               {
//                 label: 'GitHub',
//                 href: 'https://github.com/Anam-Noman?tab=repositories',
//               },
//             ],
//           },
//         ],
//         copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Curriculum. Built with Docusaurus.`,
//       },
//       prism: {
//         theme: require('prism-react-renderer').themes.github,
//         darkTheme: require('prism-react-renderer').themes.dracula,
//       },
//     }),
// };

// module.exports = config;
