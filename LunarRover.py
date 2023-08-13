import torch
import math
from datetime import datetime

from Filters.EKF_test import EKFTest

from Simulations.Extended_sysmdl import SystemModel
from Simulations.utils import DataGen,Short_Traj_Split
import Simulations.config as config
from Simulations.LunarRoverSim.parameters import x_0, P_0, m, n,\
f, h, RoverTruth, ProcessedObservations, Q_structure, R_structure

from Pipelines.Pipeline_EKF import Pipeline_EKF

from KNet.KalmanNet_nn import KalmanNetNN

from Plot import Plot_extended as Plot

print("Pipeline Start")
################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

#####################################
### Data Preprocessing & Settings ###
#####################################
args = config.general_settings()

### Data Preprocessing
# Sequence length/count
args.T = 10
args.T_test = 10
sequence_count = round(ProcessedObservations.size(1) / args.T)
# Divide all observations and ground truth into sequences
inputs = ProcessedObservations[:, :(sequence_count * args.T)].reshape(8, sequence_count, args.T).permute(1, 0, 2)
ground_truth = RoverTruth[:, :(sequence_count * args.T)].reshape(6, sequence_count, args.T).permute(1, 0, 2)
# Determine valid indices corresponding to times when >3 sats are in view
inputs_mask = (inputs != 0).all(dim=2).all(dim=1)
valid_indices = torch.nonzero(inputs_mask).squeeze()
# Remove invalid sequences
filtered_sequences = inputs[valid_indices, :, :]
filtered_sequence_count = filtered_sequences.shape[0]
filtered_ground_truth = ground_truth[valid_indices, :, :]

args.N_E = math.floor(filtered_sequence_count * 0.7)
args.N_CV = math.floor(filtered_sequence_count * 0.1)
args.N_T = math.floor(filtered_sequence_count * 0.2)

### settings for KalmanNet
args.in_mult_KNet = 40
args.out_mult_KNet = 5

### training parameters
args.use_cuda = False # use GPU or not
args.n_steps = 100
args.n_batch = 100
args.lr = 1e-4
args.wd = 1e-4
args.CompositionLoss = True
args.alpha = 1

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

chop = False
path_results = 'KNet/'

Std_Q = torch.tensor([0.0001, 0.00015, 0.0001, 0.00015, 0.0001, 0.00015, 0.0001, 0.000001])
Std_R = torch.tensor([25.087, 0.2537, 25.087, 0.2537, 25.087, 0.2537, 25.087, 0.2537])
Std_S = torch.tensor([1e3, 0, 1e3, 0, 1e3, 0, 1e3, 0])

Q = Std_Q**2 * Q_structure
R = Std_R**2 * R_structure
S = Std_S * torch.eye(m)

sys_model = SystemModel(f, Q, h, R, args.T, args.T_test, m, n)
sys_model.InitSequence(x_0, P_0)

#########################################
###  Generate and load data DT case   ###
#########################################

train_input = filtered_sequences[:args.N_E, :, :]
train_target = filtered_ground_truth[:args.N_E, :, :]

cv_input = filtered_sequences[args.N_E:(args.N_E+args.N_CV), :, :]
cv_target = filtered_ground_truth[args.N_E:(args.N_E+args.N_CV), :, :]

test_input = filtered_sequences[(args.N_E+args.N_CV):(args.N_E+args.N_CV+args.N_T), :, :]
test_target = filtered_ground_truth[(args.N_E+args.N_CV):(args.N_E+args.N_CV+args.N_T), :, :]

# train_input = ProcessedObservations[:, :args.N_E*args.T].reshape(8, args.N_E, args.T).permute(1, 0, 2)
# train_target = RoverTruth[:, :args.N_E*args.T].reshape(6, args.N_E, args.T).permute(1, 0, 2)

# cv_input = ProcessedObservations[:, args.N_E*args.T:(args.N_E+args.N_CV)*args.T].reshape(8, args.N_CV, args.T).permute(1, 0, 2)
# cv_target = RoverTruth[:, args.N_E*args.T:(args.N_E+args.N_CV)*args.T].reshape(6, args.N_CV, args.T).permute(1, 0, 2)

# test_input = ProcessedObservations[:, (args.N_E+args.N_CV)*args.T:(args.N_E+args.N_T+args.N_CV)*args.T].reshape(8, args.N_T, args.T).permute(1, 0, 2)
# test_target = RoverTruth[:, (args.N_E+args.N_CV)*args.T:(args.N_E+args.N_T+args.N_CV)*args.T].reshape(6, args.N_T, args.T).permute(1, 0, 2)

print("train_input size:",train_input.size())
print("train_target size:",train_target.size())
print("cv_input size:",cv_input.size())
print("cv_target size:",cv_target.size())
print("test_input size:",test_input.size())
print("test_target size:",test_target.size())

########################
### Evaluate Filters ###
########################
# ### Evaluate EKF full
# print("Evaluate EKF full")
# [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, EKF_KG_array, EKF_out] = EKFTest(args, sys_model, test_input, test_target)

# ### Save trajectories
# trajfolderName = 'Filters' + '/'
# DataResultName = traj_resultName[0]
# EKF_sample = torch.reshape(EKF_out[0],[1,m,args.T_test])
# target_sample = torch.reshape(test_target[0,:,:],[1,m,args.T_test])
# input_sample = torch.reshape(test_input[0,:,:],[1,n,args.T_test])
# torch.save({
#             'EKF': EKF_sample,
#             'ground_truth': target_sample,
#             'observation': input_sample,
#             }, trajfolderName+DataResultName)


##########################
### Evaluate KalmanNet ###
##########################
## Build Neural Network
print("KalmanNet start")
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model, args)
## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
KalmanNet_Pipeline.setTrainingParams(args) 
if(chop):
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, valid_indices, path_results,randomInit=True)
else:
   print("Composition Loss:",args.CompositionLoss)
   [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, valid_indices, path_results)
## Test Neural Network
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, valid_indices, path_results)

####################
### Plot results ###
####################


PlotfileName = "LunarRover_position.png"

TrajfolderName = 'Simulations/LunarRoverSim/'
DataResultName = 'Traj_LunarRover.pt'

target_sample = torch.reshape(test_target[0,:,:],[1,6,args.T_test])

torch.save({           
            'True':target_sample,                      
            'KNet': knet_out,
            }, TrajfolderName+DataResultName)

Plot = Plot(TrajfolderName, DataResultName)

titles = ["True Trajectory","KNet"]
input = [target_sample, knet_out]
Plot.plotTrajectories(input, 3, titles,TrajfolderName+PlotfileName)

# import code
# code.interact(local=locals())