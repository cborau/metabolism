// defines interactions with ECM agents and computes metabolism of species
FLAMEGPU_AGENT_FUNCTION(cell_ecm_interaction_metabolism, flamegpu::MessageArray3D, flamegpu::MessageNone) {
  //Get agent variables (agent calling the function)
  int agent_id = FLAMEGPU->getVariable<int>("id");
  float agent_x = FLAMEGPU->getVariable<float>("x");
  float agent_y = FLAMEGPU->getVariable<float>("y");
  float agent_z = FLAMEGPU->getVariable<float>("z");
  float agent_radius = FLAMEGPU->getVariable<float>("radius");
  float agent_volume = 4.0f / 3.0f * 3.1415926f * powf(agent_radius, 3); // sphere, in um^3

  const float TIME_STEP = FLAMEGPU->environment.getProperty<float>("TIME_STEP");
  const float ECM_VOXEL_VOLUME = FLAMEGPU->environment.getProperty<float>("ECM_VOXEL_VOLUME");
  
  // Agent array variables
  const uint8_t N_SPECIES = 2; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  const uint32_t ECM_POPULATION_SIZE = 1000; // WARNING: this variable must be hard coded to have the same value as the one defined in the main python function.
  auto C_SP_MACRO = FLAMEGPU->environment.getMacroProperty<float, N_SPECIES, ECM_POPULATION_SIZE>("C_SP_MACRO");
  
  float k_consumption[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_consumption[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_consumption", i);
  }
  
  float k_production[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_production[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_production", i);
  }

  float k_reaction[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    k_reaction[i] = FLAMEGPU->getVariable<float, N_SPECIES>("k_reaction", i);
  }
  
  float C_sp[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp", i);
  }

  float C_sp_sat[N_SPECIES] = {};
  for (int i = 0; i < N_SPECIES; i++) {
    C_sp_sat[i] = FLAMEGPU->getVariable<float, N_SPECIES>("C_sp_sat", i);
  }

    // Get number of agents per direction
  const int Nx = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",0);
  const int Ny = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",1);
  const int Nz = FLAMEGPU->environment.getProperty<int>("ECM_AGENTS_PER_DIR",2);
  // Get position of the boundaries
  const float COORD_BOUNDARY_X_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",0);
  const float COORD_BOUNDARY_X_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",1);
  const float COORD_BOUNDARY_Y_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",2);
  const float COORD_BOUNDARY_Y_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",3);
  const float COORD_BOUNDARY_Z_POS = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",4);
  const float COORD_BOUNDARY_Z_NEG = FLAMEGPU->environment.getProperty<float>("COORDS_BOUNDARIES",5);
  
  // transform x,y,z positions to i,j,k grid positions
  int agent_grid_i = roundf(((agent_x - COORD_BOUNDARY_X_NEG) / (COORD_BOUNDARY_X_POS - COORD_BOUNDARY_X_NEG)) * (Nx - 1));
  int agent_grid_j = roundf(((agent_y - COORD_BOUNDARY_Y_NEG) / (COORD_BOUNDARY_Y_POS - COORD_BOUNDARY_Y_NEG)) * (Ny - 1));
  int agent_grid_k = roundf(((agent_z - COORD_BOUNDARY_Z_NEG) / (COORD_BOUNDARY_Z_POS - COORD_BOUNDARY_Z_NEG)) * (Nz - 1));

  //Define message variables (agent sending the input message)
  int message_id = 0;
  int message_grid_lin_id = 0;
  float message_C_sp[N_SPECIES] = {}; // old concentration values

  printf("Cell agent %d at pos (%2.6f , %2.6f, %2.6f) reading ECM agent at grid (%d , %d, %d) \n", agent_id, agent_x, agent_y,  agent_z, agent_grid_i, agent_grid_j, agent_grid_k);

  // Reads the closest ECM agent grid_lin_id to read the corresponding C_SP_MACRO value
  // Then computes metabolism, updating both the cell calling agent and C_SP_MACRO values accordingly
  // The closest ECM agent
  const auto message = FLAMEGPU->message_in.at(agent_grid_i, agent_grid_j, agent_grid_k);
  message_id = message.getVariable<int>("id");
  message_grid_lin_id = message.getVariable<int>("grid_lin_id");
  for (int i = 0; i < N_SPECIES; i++) {

    //message_C_sp[i] = (float)C_SP_MACRO[i][message_grid_lin_id]; // read concentration of species from the MACRO variable
    
    // Backward-Euler (implicit) time integration (unconditionally stable) of:
    //   dC_cell/dt = k_production - k_consumption * C_cell
    //
    // Discretization from t^n to t^{n+1}:
    //   (C^{n+1} - C^n)/dt = k_production - k_consumption * C^{n+1}
    //
    // Solve for C^{n+1}:
    //   C^{n+1} * (1 + dt * k_consumption) = C^n + dt * k_production
    //   C^{n+1} = (C^n + dt * k_production) / (1 + dt * k_consumption)
    //
    // Using PhysiCell-style notation:
    //   c1 = dt * k_production
    //   c2 = 1 + dt * k_consumption
    //
    //   C_cell^{n+1} = (C_cell^n + c1) / c2
    //   deltaC       = C_cell^{n+1} - C_cell^n = (c1 + C_cell^n * (1 - c2)) / c2
    // ECM receives the opposite change: -deltaC
    //
    // Here:
    //   C^n             = message_C_sp[i]   (value read from the ECM grid message)
    //   C^{n+1}         = C_sp[i]           (value stored after update)
    //   dt              = TIME_STEP         (time step)
    //   k_production    = k_production[i]   (source rate for species i)
    //   k_consumption   = k_consumption[i]  (linear consumption rate for species i)

    message_C_sp[i] = message.getVariable<float, N_SPECIES>("C_sp", i);  // C^n (old)
    //printf("  -> ECM agent id %d at grid_lin_id %d has C_sp[%d] = %.6f \n", message_id, message_grid_lin_id, i+1, message_C_sp[i]);
    
    const float alpha = agent_volume / ECM_VOXEL_VOLUME; // ratio of cell volume to ECM voxel volume
    const float C_cell_old = C_sp[i]; // CHECK THIS: is this correct? or should it be message_C_sp[i]?
    const float c1 = alpha * TIME_STEP * k_production[i] * C_sp_sat[i] ;
    const float c2 = 1.0f + TIME_STEP * alpha * (k_production[i] + k_consumption[i]);

    const float C_cell_new = (C_cell_old + c1) / c2;
    const float deltaC = C_cell_new - C_cell_old;

    // Update cell concentration. Cap at zero minimum
    if (C_cell_new < 0.0f) {
      C_sp[i] = 0.0f;
    } else {
      C_sp[i] = C_cell_new;
    }
    
    // Update ECM MACRO concentration before metabolism. Cap at zero minimum
    if (message_C_sp[i] - deltaC < 0.0f) {
      C_SP_MACRO[i][message_grid_lin_id].exchange(0.0f);
    } else 
    {
      C_SP_MACRO[i][message_grid_lin_id].exchange(message_C_sp[i] - deltaC);
    }
  }

  // Compute metabolic reaction within the cell
  C_sp[0] -= TIME_STEP * k_reaction[i] * C_sp[0]; // species 0 is consumed
  C_sp[1] += TIME_STEP * k_reaction[i] * C_sp[0]; // species 1 is produced accordingly

  for (int i = 0; i < N_SPECIES; i++) {
    FLAMEGPU->setVariable<float, N_SPECIES>("C_sp", i, C_sp[i]);
  }

  return flamegpu::ALIVE;
}