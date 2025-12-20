import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';
import Layout from '@theme/Layout';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroText}>
            <h1 className="hero__title">{siteConfig.title}</h1>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <div className={styles.description}>
              <p>This comprehensive curriculum bridges the gap between digital AI models and physical robotic bodies, providing you with the knowledge and practical skills to create embodied intelligence systems.</p>
            </div>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro/">
                Get Started
              </Link>
            </div>
          </div>
          <div className={styles.heroDetails}>
            <div className={styles.detailCard}>
              <h3>🎯 Curriculum Structure</h3>
              <ul>
                <li>Module 1: ROS 2 (Robotic Nervous System)</li>
                <li>Module 2: Digital Twin (Gazebo & Unity)</li>
                <li>Module 3: AI-Robot Brain (NVIDIA Isaac)</li>
                <li>Module 4: Vision-Language-Action (VLA)</li>
              </ul>
            </div>
            <div className={styles.detailCard}>
              <h3>📚 Learning Approach</h3>
              <ul>
                <li>Hands-on projects after each module</li>
                <li>13-week structured curriculum</li>
                <li>Simulation-based development</li>
                <li>Real-world robotics applications</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Curriculum">
      <HomepageHeader />
    </Layout>
  );
}