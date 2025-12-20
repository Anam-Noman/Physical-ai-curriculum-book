import React from 'react';
import clsx from 'clsx';
import styles from './ROS2Communication.module.css';

// Define the SVG diagram content for ROS 2 communication patterns
function ROS2CommunicationSvg({ className, ...props }) {
  return (
    <div className={clsx('col col--12', className)} {...props}>
      <svg width="100%" height="500" viewBox="0 0 900 500" xmlns="http://www.w3.org/2000/svg">
        {/* Background */}
        <rect width="100%" height="100%" fill="#f9fafb" rx="10" />
        
        {/* Title */}
        <text x="450" y="40" textAnchor="middle" fontWeight="bold" fontSize="20" fill="#1f2937">ROS 2 Communication Patterns</text>
        
        {/* Publisher Node */}
        <rect x="50" y="80" width="150" height="80" fill="#dbeafe" stroke="#3b82f6" strokeWidth="2" rx="10" />
        <text x="125" y="115" textAnchor="middle" fontWeight="bold" fontSize="14">Publisher Node</text>
        <text x="125" y="135" textAnchor="middle" fontSize="12">Talker</text>
        
        {/* Subscriber Node */}
        <rect x="50" y="200" width="150" height="80" fill="#dbeafe" stroke="#3b82f6" strokeWidth="2" rx="10" />
        <text x="125" y="235" textAnchor="middle" fontWeight="bold" fontSize="14">Subscriber Node</text>
        <text x="125" y="255" textAnchor="middle" fontSize="12">Listener</text>
        
        {/* Service Client Node */}
        <rect x="50" y="320" width="150" height="80" fill="#fed7e2" stroke="#ec4899" strokeWidth="2" rx="10" />
        <text x="125" y="355" textAnchor="middle" fontWeight="bold" fontSize="14">Service Client</text>
        <text x="125" y="375" textAnchor="middle" fontSize="12">AddTwoInts</text>
        
        {/* Service Server Node */}
        <rect x="50" y="420" width="150" height="80" fill="#fed7e2" stroke="#ec4899" strokeWidth="2" rx="10" />
        <text x="125" y="455" textAnchor="middle" fontWeight="bold" fontSize="14">Service Server</text>
        <text x="125" y="475" textAnchor="middle" fontSize="12">AddTwoInts</text>
        
        {/* Topic: chatter */}
        <rect x="300" y="80" width="120" height="80" fill="#dcfce7" stroke="#22c55e" strokeWidth="2" rx="5" />
        <text x="360" y="115" textAnchor="middle" fontWeight="bold" fontSize="14">Topic</text>
        <text x="360" y="135" textAnchor="middle" fontSize="12">chatter</text>
        
        {/* Arrow from Publisher to Topic */}
        <line x1="200" y1="120" x2="300" y2="120" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Arrow from Topic to Subscriber */}
        <line x1="420" y1="120" x2="520" y2="120" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Subscriber Node (right side) */}
        <rect x="520" y="80" width="150" height="80" fill="#dbeafe" stroke="#3b82f6" strokeWidth="2" rx="10" />
        <text x="595" y="115" textAnchor="middle" fontWeight="bold" fontSize="14">Subscriber</text>
        <text x="595" y="135" textAnchor="middle" fontSize="12">Listener</text>
        
        {/* Service Request Arrow */}
        <line x1="200" y1="360" x2="350" y2="360" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <text x="275" y="350" textAnchor="middle" fontSize="10">Request</text>
        
        {/* Service Response Arrow */}
        <line x1="550" y1="360" x2="400" y2="360" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        <text x="475" y="350" textAnchor="middle" fontSize="10">Response</text>
        
        {/* Service Interface */}
        <rect x="350" y="320" width="50" height="80" fill="#fef3c7" stroke="#f59e0b" strokeWidth="2" rx="5" />
        <text x="375" y="355" textAnchor="middle" fontWeight="bold" fontSize="10">Service</text>
        <text x="375" y="370" textAnchor="middle" fontSize="10">Interface</text>
        
        {/* DDS Middleware */}
        <rect x="250" y="220" width="400" height="60" fill="#e0e7ff" stroke="#6366f1" strokeWidth="2" rx="5" />
        <text x="450" y="250" textAnchor="middle" fontWeight="bold" fontSize="14">DDS Middleware</text>
        
        {/* Arrow from Publisher to DDS */}
        <line x1="200" y1="110" x2="250" y2="220" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Arrow from DDS to Client */}
        <line x1="250" y1="250" x2="200" y2="340" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Arrow from Server to DDS */}
        <line x1="200" y1="460" x2="250" y2="280" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Arrow from DDS to Subscriber */}
        <line x1="650" y1="110" x2="650" y2="220" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
        {/* Arrow from DDS to Publisher */}
        <line x1="650" y1="280" x2="650" y2="460" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowhead)" />
        
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

export default function ROS2Communication() {
  return (
    <section className={styles.ros2Communication}>
      <div className="container">
        <div className="row">
          <ROS2CommunicationSvg />
          <div className="col col--12 padding-horiz--md">
            <h3 className="text--center padding-top--md">
              ROS 2 Communication Patterns Overview
            </h3>
            <p className="text--center">
              This diagram illustrates the three primary communication patterns in ROS 2: 
              topics (publish/subscribe), services (request/response), and the underlying DDS middleware.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}