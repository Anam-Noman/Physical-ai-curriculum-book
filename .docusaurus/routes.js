import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/physical-ai-curriculum-book/__docusaurus/debug',
    component: ComponentCreator('/physical-ai-curriculum-book/__docusaurus/debug', 'd99'),
    exact: true
  },
  {
    path: '/physical-ai-curriculum-book/__docusaurus/debug/config',
    component: ComponentCreator('/physical-ai-curriculum-book/__docusaurus/debug/config', 'ab0'),
    exact: true
  },
  {
    path: '/physical-ai-curriculum-book/__docusaurus/debug/content',
    component: ComponentCreator('/physical-ai-curriculum-book/__docusaurus/debug/content', '7b4'),
    exact: true
  },
  {
    path: '/physical-ai-curriculum-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/physical-ai-curriculum-book/__docusaurus/debug/globalData', '692'),
    exact: true
  },
  {
    path: '/physical-ai-curriculum-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/physical-ai-curriculum-book/__docusaurus/debug/metadata', '733'),
    exact: true
  },
  {
    path: '/physical-ai-curriculum-book/__docusaurus/debug/registry',
    component: ComponentCreator('/physical-ai-curriculum-book/__docusaurus/debug/registry', '115'),
    exact: true
  },
  {
    path: '/physical-ai-curriculum-book/__docusaurus/debug/routes',
    component: ComponentCreator('/physical-ai-curriculum-book/__docusaurus/debug/routes', 'f05'),
    exact: true
  },
  {
    path: '/physical-ai-curriculum-book/docs',
    component: ComponentCreator('/physical-ai-curriculum-book/docs', 'e18'),
    routes: [
      {
        path: '/physical-ai-curriculum-book/docs',
        component: ComponentCreator('/physical-ai-curriculum-book/docs', 'f44'),
        routes: [
          {
            path: '/physical-ai-curriculum-book/docs',
            component: ComponentCreator('/physical-ai-curriculum-book/docs', 'e61'),
            routes: [
              {
                path: '/physical-ai-curriculum-book/docs/intro/',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/intro/', '61a'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/intro/prerequisites',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/intro/prerequisites', 'a2c'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/intro/setup',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/intro/setup', '256'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/', '43f'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/physical-ai-concepts',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/physical-ai-concepts', '456'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/summary',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/summary', '578'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/week-1-2/embodied-intelligence',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/week-1-2/embodied-intelligence', 'f17'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/week-1-2/foundations',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/week-1-2/foundations', '4d0'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/architecture',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/architecture', 'f45'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/nodes-topics',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/nodes-topics', 'bc1'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/rclpy-bridge',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/rclpy-bridge', 'cbb'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/urdf-modeling',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-1-ros2/week-3-5/urdf-modeling', '760'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/', '04d'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/digital-twin-concept',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/digital-twin-concept', '269'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/summary',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/summary', 'cf1'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/gazebo-setup',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/gazebo-setup', 'e5c'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/physics-simulation',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/physics-simulation', '26b'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/sensor-simulation',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/sensor-simulation', '1eb'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/unity-visualization',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/unity-visualization', '7bf'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/urdf-sdf',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-2-digital-twin/week-6-7/urdf-sdf', '24b'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/', '8ec'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/sim-to-real',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/sim-to-real', '54d'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/summary',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/summary', 'ee3'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/hri-design',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/hri-design', '0ab'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/humanoid-dynamics',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/humanoid-dynamics', 'ce9'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/locomotion-balance',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/locomotion-balance', '770'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/manipulation-grasping',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-11-12/manipulation-grasping', '8c8'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/isaac-platform',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/isaac-platform', 'd1d'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/nav2-planning',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/nav2-planning', 'c2f'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/perception-pipelines',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/perception-pipelines', '1d6'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/synthetic-data',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/synthetic-data', 'ad2'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/vslam-localization',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-3-ai-brain/week-8-10/vslam-localization', '19c'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-4-vla/',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-4-vla/', 'e49'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/module-4-vla/week-13/gpt-integration',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/module-4-vla/week-13/gpt-integration', 'ef2'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/tutorials/basic-ros2-tutorial',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/tutorials/basic-ros2-tutorial', 'e26'),
                exact: true,
                sidebar: "curriculumSidebar"
              },
              {
                path: '/physical-ai-curriculum-book/docs/tutorials/simulation-tutorial',
                component: ComponentCreator('/physical-ai-curriculum-book/docs/tutorials/simulation-tutorial', '76d'),
                exact: true,
                sidebar: "curriculumSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/physical-ai-curriculum-book/',
    component: ComponentCreator('/physical-ai-curriculum-book/', '11b'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
