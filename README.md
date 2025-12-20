# Physical AI & Humanoid Robotics Curriculum Book

This repository contains a comprehensive technical book on Physical AI and Humanoid Robotics, designed as a Docusaurus-based educational resource covering 4 core modules:

1. **ROS 2 (The Robotic Nervous System)**: Middleware for robot control and communication
2. **Digital Twin (Gazebo & Unity)**: Physics simulation and environment modeling
3. **AI-Robot Brain (NVIDIA Isaac)**: Perception, learning, and navigation
4. **Vision-Language-Action (VLA)**: Natural human-robot interaction

## Curriculum Overview

The book follows a 13-week curriculum structure with detailed learning outcomes:
- Weeks 1-2: Physical AI foundations and embodied intelligence
- Weeks 3-5: ROS 2 fundamentals and architecture
- Weeks 6-7: Digital Twin and simulation environments
- Weeks 8-10: NVIDIA Isaac and perception systems
- Weeks 11-12: Humanoid kinematics and dynamics
- Week 13: Vision-Language-Action integration

## Target Audience

- Advanced Computer Science students
- Robotics learners
- AI engineers preparing for Physical AI and humanoid robotics projects

## Technical Requirements

- Node.js 18+ 
- Docusaurus 3.x
- ROS 2 (Humble Hawksbill)
- NVIDIA Isaac Sim (optional)
- Gazebo simulation environment (optional)
- Unity 2023.2 LTS (optional)

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Open [http://localhost:3000](http://localhost:3000) to view the book in your browser.

## Build for Production

To build the book for deployment:

```bash
npm run build
```

The static files will be generated in the `build` directory, ready for deployment to GitHub Pages.

## Contributing

This project follows a spec-driven development approach. All content changes should be traced back to the feature specifications in the `specs/` directory.