import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import styles from './GazeboSimulationViewer.module.css';

// Define the properties for the Gazebo Simulation Viewer component
const GazeboSimulationBlock = {
  title: 'Gazebo Simulation Environment',
  description: (
    <>
      Interactive visualization of a simulated robot environment with realistic physics, 
      sensors, and robot models using Gazebo simulation.
    </>
  ),
};

// Function to render the Gazebo simulation viewer
function GazeboSimulationViewer({ className, ...props }) {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Simulate loading of the simulation viewer
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={clsx('col col--12', className)} {...props}>
      <div className={styles.simulationContainer}>
        <div className={styles.simulationHeader}>
          <h3>{GazeboSimulationBlock.title}</h3>
          <div className={styles.simulationControls}>
            <button className={styles.controlButton}>Play</button>
            <button className={styles.controlButton}>Pause</button>
            <button className={styles.controlButton}>Reset</button>
          </div>
        </div>
        
        <div className={styles.simulationView}>
          {isLoaded ? (
            <div className={styles.simulationContent}>
              <div className={styles.robotModel}>
                <div className={styles.robotBase}></div>
                <div className={styles.robotWheel}></div>
                <div className={styles.robotWheel}></div>
                <div className={styles.robotSensor}></div>
              </div>
              
              <div className={styles.environment}>
                <div className={styles.groundPlane}></div>
                <div className={styles.wall}></div>
                <div className={styles.wall}></div>
                <div className={styles.wall}></div>
                <div className={styles.wall}></div>
                <div className={styles.obstacle}></div>
              </div>
            </div>
          ) : (
            <div className={styles.loading}>
              <div className={styles.spinner}></div>
              <p>Loading simulation environment...</p>
            </div>
          )}
        </div>
        
        <div className={styles.simulationInfo}>
          <div className={styles.infoItem}>
            <span className={styles.infoLabel}>Physics Engine:</span>
            <span className={styles.infoValue}>ODE</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoLabel}>Gravity:</span>
            <span className={styles.infoValue}>-9.8 m/s²</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoLabel}>Sensors:</span>
            <span className={styles.infoValue}>Camera, LiDAR, IMU</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Function to render a Unity visualization viewer
function UnityVisualizationViewer({ className, ...props }) {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Simulate loading of the Unity viewer
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 1500);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={clsx('col col--12', className)} {...props}>
      <div className={clsx(styles.simulationContainer, styles.unityContainer)}>
        <div className={styles.simulationHeader}>
          <h3>Unity Visualization Environment</h3>
          <div className={styles.simulationControls}>
            <button className={styles.controlButton}>Play</button>
            <button className={styles.controlButton}>Pause</button>
          </div>
        </div>
        
        <div className={styles.simulationView}>
          {isLoaded ? (
            <div className={styles.simulationContent}>
              <div className={styles.unityScene}>
                <div className={styles.robotModelUnity}>
                  <div className={styles.robotBaseUnity}></div>
                  <div className={styles.robotCameraUnity}></div>
                </div>
                
                <div className={styles.unityEnvironment}>
                  <div className={styles.unityGround}></div>
                  <div className={styles.unityWall}></div>
                  <div className={styles.unityWall}></div>
                  <div className={styles.unityFurniture}></div>
                </div>
                
                <div className={styles.visualizationOverlay}>
                  <div className={styles.pathVisualization}></div>
                  <div className={styles.fovVisualization}></div>
                </div>
              </div>
            </div>
          ) : (
            <div className={styles.loading}>
              <div className={clsx(styles.spinner, styles.unitySpinner)}></div>
              <p>Loading Unity visualization...</p>
            </div>
          )}
        </div>
        
        <div className={styles.simulationInfo}>
          <div className={styles.infoItem}>
            <span className={styles.infoLabel}>Renderer:</span>
            <span className={styles.infoValue}>High-fidelity</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoLabel}>Visualization:</span>
            <span className={styles.infoValue}>Photorealistic</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoLabel}>Interaction:</span>
            <span className={styles.infoValue}>Human-robot interface</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main export function
export default function SimulationViewers() {
  return (
    <section className={styles.simulationViewers}>
      <div className="container">
        <div className="row">
          <GazeboSimulationViewer />
          <div className="col col--12 padding-horiz--md">
            <p className="text--center padding-top--md">
              {GazeboSimulationBlock.description}
            </p>
          </div>
        </div>
        
        <div className="row padding-top--lg">
          <UnityVisualizationViewer />
          <div className="col col--12 padding-horiz--md">
            <p className="text--center padding-top--md">
              Unity provides high-fidelity visualization and human-robot interaction capabilities 
              for advanced Physical AI applications.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}