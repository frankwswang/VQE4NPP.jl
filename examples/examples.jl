using VQE4NPP

s = 100 # Define the sum of the elements inside the target set. 
n = 5 # Define the number of elements inside the target set.
d = 6 # Define the depth of the differentiable circuit.

set = RandIntNumSet(n,s) # Create the target number set based on configurartions
VQEtrain(set, depth = d) # Train the VQE for the corresponding Hamiltonian of the target set.