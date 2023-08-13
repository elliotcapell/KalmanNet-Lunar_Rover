import torch
import numpy as np
import math
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from torch import autograd

###################
### Import Data ###
###################
RoverTruth = torch.load('Simulations/LunarRoverSim/data/RoverTruth.pt')
SkippedSatellite = torch.load('Simulations/LunarRoverSim/data/SkippedSatellite.pt')
ProcessedObservations = torch.load('Simulations/LunarRoverSim/data/ProcessedObservations.pt')
SatX = torch.load('Simulations/LunarRoverSim/data/SatX.pt')
SatY = torch.load('Simulations/LunarRoverSim/data/SatY.pt')
SatZ = torch.load('Simulations/LunarRoverSim/data/SatZ.pt')
SatVx = torch.load('Simulations/LunarRoverSim/data/SatVx.pt')
SatVy = torch.load('Simulations/LunarRoverSim/data/SatVy.pt')
SatVz = torch.load('Simulations/LunarRoverSim/data/SatVz.pt')

#########################
### Design Parameters ###
#########################
T = 10 # Sequence length
m = 8
n = 8
variance = 0
x_0 = torch.tensor([
    13785.8859115925,
    0.106463063435197,
    -12616.9930942748,
    0.0694525864003876,
    -1736366.32259110,
    -0.0843550264298861,
    88.8395631757642,
    -0.114707010696915]).reshape(-1, 1)
P_0 = torch.tensor([100**2, 0**2, 100**2, 0**2, 100**2, 0**2, 100**2, 0.1**2]) * torch.eye(n)

PredMeasurements = torch.zeros((1, n+2, 1))

Q_structure = torch.eye(m)
R_structure = torch.eye(n)

### Decimation
dt = 1

######################################################
### Constant Velocity State evolution function f   ###
######################################################

def f(x):
    Fsub = np.array([[1, dt], [0, 1]], dtype=np.float32)
    F = torch.tensor(np.kron(np.eye(4), Fsub), dtype=torch.float32).unsqueeze(0).repeat(x.size(0), 1, 1)
    return torch.bmm(F, x)

##################################################
### Pseudorange/Rate Observation function h    ###
##################################################

def h(State_est, BatchIndexes, t):

    PredMeasurements = torch.zeros((BatchIndexes.numel(), 8, 1))
    BatchIndexes = BatchIndexes * T + t 
    i = 0

    for k in BatchIndexes:
        k = int(k.item())
        b = 0
        for j in range(5):

            # Skips when a satellite is occulted by the moon or based on GDOP
            if j == SkippedSatellite[0, k] or SkippedSatellite[0, k] == -1:
                continue
            
            # Calculating magnitude of range vector
            Di = torch.sqrt((SatX[k, j] - State_est[i, 0])**2 + (SatY[k, j] - State_est[i, 2])**2 + (SatZ[k, j] - State_est[i, 4])**2)

            # Calculating relative velocity vector
            Vi = -((SatVx[k, j] - State_est[i, 1]) * -(SatX[k,j]-State_est[i, 0])/Di + (SatVy[k, j] - State_est[i, 3]) * -(SatY[k,j]-State_est[i, 2])/Di + (SatVz[k, j] - State_est[i, 5]) * -(SatZ[k,j]-State_est[i, 4])/Di)

            # Predicted Measurement
            PredMeasurements[i, b, 0] = Di + State_est[i, 6]
            PredMeasurements[i, b + 1, 0] = Vi + State_est[i, 7]
            b = b + 2

            # print("Current k, j, b, t:", k, j, b-2, t)
            # print("Di:", Di.item())
            # print("Vi:", Vi.item())
            # print('SatX:', SatX[k, j])
            # print('SatY:', SatY[k, j])
            # print('SatZ:', SatZ[k, j])
            # print('SatVx:', SatVx[k, j])
            # print('SatVy:', SatVy[k, j])
            # print('SatVz:', SatVz[k, j])
            # print('Value:', State_est[i, 0], State_est[i, 2], State_est[i, 4], (SatX[k, j] - State_est[i, 0])**2)

        i = i + 1

    return PredMeasurements