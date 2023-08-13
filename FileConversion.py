import pandas as pd
import torch
import numpy as np

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/RoverTruth.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/RoverTruth.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/SkippedSatellite.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/SkippedSatellite.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/ProcessedObservations.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/ProcessedObservations.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/SatX.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/SatX.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/SatY.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/SatY.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/SatZ.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/SatZ.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/SatVx.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/SatVx.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/SatVy.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/SatVy.pt')

####################################################################

# Read CSV file using Pandas
csv_file_path = 'Simulations/LunarRoverSim/rawdata/SatVz.txt'
data_frame = pd.read_csv(csv_file_path, header=None)

# Convert to NumPy array
numpy_array = data_frame.values

# Convert to PyTorch tensor
pytorch_tensor = torch.tensor(numpy_array, dtype=torch.float32)

# Save tensor to .pt file
torch.save(pytorch_tensor, 'Simulations/LunarRoverSim/data/SatVz.pt')