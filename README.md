# Task 9: Simulation Environment

## Environment Setup
To establish the simulation environment, the required repository was cloned from GitHub. The repository, named **Y2B-2023-OT2_Twin**, contains all the necessary resources for the simulation. Additionally, the **Visual Studio C++ Toolkit** was installed to facilitate the integration and functionality of PyBullet. A script named `task_9.py` was developed to initiate and execute the simulation.

## Prerequisites
Before running the simulation, ensure the following dependencies are installed and set up correctly:

- **PyBullet**: Version 3.2.6
- **NumPy**: Version 1.26.4

Additionally, the following resources are required from the cloned repository:
- Files from the **Y2B-2023-OT2_Twin** repository.

## Simulation Workflow
The script `task_9.py` is responsible for running the simulation. To ensure smooth execution, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SomaGergo233650/RL.git
   ```

2. Install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify that the Visual Studio C++ Toolkit is installed and properly configured for your environment.

4. Run the simulation by executing the script:
   ```bash
   python task_9.py
   ```

## Determining the Operational Envelope
The operational envelope represents the boundaries within which the agent can operate in the simulation environment. This was determined by:

1. Applying various velocity combinations.
2. Monitoring and logging the position of the agent at each timestep.
3. Identifying the corner coordinates of the operational space.
4. Implementing a navigation path to ensure the agent follows these boundaries effectively.

This process ensures accurate and consistent agent movement within the simulated environment.

## Features
- Accurate visualization of the operational envelope.
- Seamless integration with PyBullet for realistic simulation dynamics.
- User-friendly script to initialize and run the simulation.

## Repository Link
[Y2B-2023-OT2_Twin Repository](<https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git>)


