import numpy as np
import pickle

# Generate random network of 10 nodes within a cubical domain [-0.5, 0.5]
num_nodes = 10
nodes = np.random.uniform(low=-0.5, high=0.5, size=(num_nodes, 3))

# Create random connectivity, ensuring each node connects to at least one other node
connectivity = {}
max_connectivity = 8  # Maximum number of connections for each node
for i in range(num_nodes):
    # Randomly choose other nodes to connect to (excluding self-connections)
    num_connections = np.random.randint(1, max_connectivity + 1)
    connections = np.random.choice([j for j in range(num_nodes) if j != i], size=num_connections, replace=False)
    # Ensure the connectivity list has -1 for unused slots
    connectivity[i] = list(connections) + [-1] * (max_connectivity - len(connections))

# Save to pickle file
with open('network_3d_fake.pkl', 'wb') as f:
    pickle.dump({'node_coords': nodes, 'connectivity': connectivity}, f)

print("Network saved to 'network_3d_fake.pkl'")
