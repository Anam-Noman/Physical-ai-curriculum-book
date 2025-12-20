import React from 'react';
import clsx from 'clsx';
import styles from './Citation.module.css';

// This component renders an APA-style citation
function Citation({ authors, year, title, source, link }) {
  return (
    <div className={clsx('col col--12', styles.citationBlock)}>
      <p className={styles.citation}>
        {authors} ({year}). <em>{title}</em>. {source}. 
        {link && <a href={link} target="_blank" rel="noopener noreferrer"> [Link]</a>}
      </p>
    </div>
  );
}

// Define an example citation list component
export default function CitationList() {
  const citations = [
    {
      authors: "Fox, G., & Company, Inc.",
      year: "2023",
      title: "ROS 2 Documentation and User Guide",
      source: "Robot Operating System",
      link: "https://docs.ros.org/en/humble/"
    },
    {
      authors: "NVIDIA Corporation",
      year: "2023",
      title: "Isaac Sim User Guide",
      source: "NVIDIA Developer Documentation",
      link: "https://docs.nvidia.com/isaac/isaac_sim/"
    },
    {
      authors: "Open Robotics",
      year: "2023",
      title: "Gazebo Simulation Engine Documentation",
      source: "Gazebo Documentation",
      link: "https://gazebosim.org/docs/"
    },
    {
      authors: "Unity Technologies",
      year: "2023",
      title: "Unity 2023.2 LTS User Manual",
      source: "Unity Documentation",
      link: "https://docs.unity3d.com/2023.2/Documentation/Manual/"
    }
  ];

  return (
    <section className={styles.references}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2>References</h2>
            <div className="padding-horiz--md">
              {citations.map((citation, index) => (
                <Citation 
                  key={index}
                  authors={citation.authors}
                  year={citation.year}
                  title={citation.title}
                  source={citation.source}
                  link={citation.link}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}