# Lithium-ion Battery Cyclic Ageing Model using Universal Differential Equations

This repository contains the Julia implementation of a Lithium-ion Battery Cyclic Ageing Model developed using **Universal Differential Equations (UDEs)**. The model is based on the research article:

> **"Enhancing the Modelling of Battery Degradation Mechanisms in Physics-based Models using Scientific Machine Learning"**  
> *by Jishnu Ayyangatu Kuzhiyil, Theodoros Damoulas, Ferran Brosa Planella, James Marco, and W. Dhammika Widanage.*

## Model Overview

This computational framework simulates the cyclic ageing behavior of Lithium-ion batteries using two distinct modelling approaches:

### 1. **Physics-Based Model**

- Employs a **Thermal Single Particle Model (SPM)** as the celectrochemical model.
- Incorporates physics-based goverining equations for the following degradation mechanisms,
  - Solid Electrolyte Interphase (SEI) growth
  - Pore blockage
  - Lithium plating
  - Particle cracking and SEI formation on cracks
  - Mechanical damage

### 2. **UDE-Based Model** 

- Builds upon the physics-based framework.
- Retains most of the original physics-based model structure.
- Replaces the governing equations for mechanical damage at both electrodes with trained data-driven models.

## ⚙️ Installation & Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/JishnuKuzhiyil/UDE-cyclic-ageing-model.git
cd UDE-cyclic-ageing-model
```

### Step 2: Set Up the Environment

Open Julia and install the required packages:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Step 3: Run the Models

Open either of the following scripts in your Julia environment:

- `Physics_model.jl` (for the physics-based model)
- `UDE_model.jl` (for the hybrid UDE model)

### Step 4: Configure Simulation Scenarios

- Set the parameter `Condition_index` (integer values from 1 to 21) to simulate different ageing scenarios.
- Refer to the provided array `condition_names` for descriptions of each ageing condition.

### Step 5: Execute the Simulation

Run the chosen script to perform simulations and view results.

## 📊 Output

Upon execution, the script will produce plots comparing model predictions against experimental measurements for:

- Capacity fade
- Internal resistance increase
- Loss of Active Material in Negative Electrode (LAM-NE)
- Loss of Active Material in Positive Electrode (LAM-PE)
- Loss of Lithium Inventory (LLI)

