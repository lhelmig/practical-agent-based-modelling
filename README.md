# Traffic Flow Modeling: ASEP and Nagel-Schreckenberg Models

## Overview

This project involves the simulation and analysis of traffic flow models using the Asymmetric Simple Exclusion Process (ASEP) and the Nagel-Schreckenberg (NaSch) model. These models help in understanding the dynamics of vehicular traffic under different conditions and boundary settings. The project also explores the Velocity-Dependent Randomization (VDR) model as an extension to the NaSch model to account for different traffic behaviors.

## Objectives

1. **ASEP Model**: Simulate traffic flow using the Asymmetric Simple Exclusion Process with both periodic and open boundary conditions. Different update schemes are implemented to observe their effects on traffic dynamics.
2. **Nagel-Schreckenberg Model**: Implement and analyze the Nagel-Schreckenberg model, which introduces acceleration, deceleration, and randomization phases to simulate real-world traffic flow, including the formation of traffic jams.
3. **Velocity-Dependent Randomization (VDR) Model**: Extend the NaSch model by incorporating velocity-dependent randomization to better capture the effects of speed variability on traffic dynamics.

## Key Components

- **`asep.py`**: Contains the implementation of the ASEP model with various update schemes:
  - **Parallel Update**: All particles attempt to move simultaneously.
  - **Sequential Update**: Particles are updated one by one in a random or fixed order.
  - **Shuffle Update**: Similar to sequential but the order is randomized in each step.
  - **Random Sequential Update**: Particles are updated randomly, and only a fraction of the particles may move in each step.
  - Handles both periodic and open boundary conditions.

- **`na_sch_model.py`**: Implements the Nagel-Schreckenberg model and its extensions:
  - **Acceleration Phase**: All vehicles attempt to increase their speed.
  - **Deceleration Phase**: Vehicles slow down if there is a vehicle ahead.
  - **Randomization Phase**: Vehicles randomly slow down to simulate variability in driver behavior.
  - **Movement Phase**: Vehicles are moved according to their updated velocities.
  - Supports Velocity-Dependent Randomization (VDR) for more realistic traffic flow modeling.

## Theoretical Background

1. **ASEP Model**: The Asymmetric Simple Exclusion Process (ASEP) is a simple model for describing particle hopping dynamics on a lattice, which can represent vehicles on a one-lane road. The hopping rate is determined by the density of vehicles and a hopping probability \( q \). Different update methods are used to simulate various types of traffic flows and analyze the impact on flux and jam formation.

2. **Nagel-Schreckenberg Model**: The NaSch model is a cellular automaton-based traffic model that accounts for driver behaviors like acceleration, deceleration, and random slowing down. The model is extended with VDR to include more complex driver reactions depending on their velocities.

## How to Run

1. **Requirements**:
   - Python 3.x
   - Numpy
   - Matplotlib
   - Numba

2. **Execution**:
   - To simulate the ASEP model with different update schemes:
     ```bash
     python asep.py
     ```
   - To run the Nagel-Schreckenberg model and its extensions:
     ```bash
     python na_sch_model.py
     ```

## Results

- **ASEP Model**:
  - The fundamental diagrams for different update schemes show how traffic flux depends on vehicle density and hopping probability \( q \).
  - Simulations with periodic and open boundary conditions illustrate different traffic phases, including low-density, high-density, and high-current phases.

- **Nagel-Schreckenberg Model**:
  - Space-time diagrams show the formation of traffic jams and the impact of random deceleration.
  - Fundamental diagrams demonstrate the transition between free flow and congested states, highlighting the effect of the randomization parameter \( p \) on traffic dynamics.

## Figures and Analysis

- **Figure 1 to 14**: Display various aspects of the ASEP model, including fundamental diagrams, density profiles, and the impact of different update methods.
- **Figure 15 to 24**: Illustrate the behavior of the Nagel-Schreckenberg and VDR models, showcasing traffic jam formation, fundamental diagrams, and convergence of mean velocity.

## Conclusion

This project successfully implements and analyzes multiple traffic flow models to study their dynamics under different conditions. The ASEP model provides a basic framework to understand traffic flow, while the Nagel-Schreckenberg model introduces more realistic traffic behaviors. The VDR model further enhances the NaSch model by incorporating variability in driver speed, leading to more nuanced traffic flow patterns.

## Future Work

- Extend the models to multi-lane traffic scenarios.
- Incorporate real-world traffic data for model validation and calibration.
- Explore the effects of more complex boundary conditions and traffic rules.

## Contact

For any questions or contributions, please contact the project author.
