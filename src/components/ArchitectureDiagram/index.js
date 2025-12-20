import React from 'react';
import clsx from 'clsx';
import styles from './ArchitectureDiagram.module.css';

// Define the properties for the ArchitectureDiagram component
const ArchitectureDiagramBlock = {
  title: 'Physical AI Architecture',
  description: (
    <>
      The Physical AI system architecture connecting digital AI models with physical robotic bodies
      through simulation, perception, planning, and action using ROS 2, Gazebo, NVIDIA Isaac, and VLA systems.
    </>
  ),
};

// Define the SVG diagram content
function ArchitectureDiagramSvg({ className, ...props }) {
  return (
    <div className={clsx('col col--12', className)} {...props}>
      <svg width="100%" height="400" viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
        {/* Background */}
        <rect width="100%" height="100%" fill="#f5f5f5" rx="10" />
        
        {/* Digital AI Layer */}
        <rect x="50" y="50" width="200" height="80" fill="#e0f2fe" stroke="#0ea5e9" strokeWidth="2" rx="5" />
        <text x="150" y="85" textAnchor="middle" fontWeight="bold" fontSize="14">Digital AI Models</text>
        <text x="150" y="105" textAnchor="middle" fontSize="12">LLMs, Perception Models</text>
        
        {/* ROS 2 Layer */}
        <rect x="300" y="50" width="200" height="80" fill="#fee2e2" stroke="#f87171" strokeWidth="2" rx="5" />
        <text x="400" y="85" textAnchor="middle" fontWeight="bold" fontSize="14">ROS 2 Middleware</text>
        <text x="400" y="105" textAnchor="middle" fontSize="12">Nodes, Topics, Actions</text>
        
        {/* Physical Robot Layer */}
        <rect x="550" y="50" width="200" height="80" fill="#dcfce7" stroke="#4ade80" strokeWidth="2" rx="5" />
        <text x="650" y="85" textAnchor="middle" fontWeight="bold" fontSize="14">Physical Robot</text>
        <text x="650" y="105" textAnchor="middle" fontSize="12">Humanoid, Sensors, Actuators</text>
        
        {/* Simulation Layer */}
        <rect x="175" y="160" width="450" height="80" fill="#ede9fe" stroke="#a78bfa" strokeWidth="2" rx="5" />
        <text x="400" y="195" textAnchor="middle" fontWeight="bold" fontSize="14">Simulation Environment</text>
        <text x="400" y="215" textAnchor="middle" fontSize="12">Gazebo, Isaac Sim, Unity</text>
        
        {/* Directional arrows */}
        <line x1="250" y1="90" x2="300" y2="90" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="500" y1="90" x2="550" y2="90" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="400" y1="130" x2="400" y2="160" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="400" y1="240" x2="400" y2="270" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <line x1="550" y1="270" x2="250" y2="270" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* VLA System */}
        <rect x="300" y="270" width="200" height="80" fill="#fffbeb" stroke="#fbbf24" strokeWidth="2" rx="5" />
        <text x="400" y="305" textAnchor="middle" fontWeight="bold" fontSize="14">VLA System</text>
        <text x="400" y="325" textAnchor="middle" fontSize="12">Vision-Language-Action</text>
        
        {/* Arrowhead marker definition */}
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" 
            refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#64748b" />
          </marker>
        </defs>
      </svg>
    </div>
  );
}

export default function ArchitectureDiagram() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          <ArchitectureDiagramSvg title={ArchitectureDiagramBlock.title} />
          <div className="col col--12 padding-horiz--md">
            <p className="text--center padding-top--md">
              {ArchitectureDiagramBlock.description}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}