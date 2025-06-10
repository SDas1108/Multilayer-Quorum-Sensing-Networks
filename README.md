Readme file
This repository contains MATLAB code for simulating and analyzing quorum sensing (QS) in heterogeneous bacterial communities using the perceptron model. The framework explores how signal-mediated communication and structural perturbations influence network activity, stability, and inter-layer information flow. 
How to Use
To run any of the simulations, make sure you download the repository and it contains neuralnetworkobject.m, structural_perturbation.m, Activation_statistics.m, Perturbation_statistics.m.
Requirements
MATLAB R2021a or later


Authors
Soumya Das, Enes Haximhali : Boedicker Lab, USC 
License
USC Licence 
Repository Structure
Neuralnetworkobject.m : Class for QS-based neural network objects
structural_perturbation.m : Class for applying and analyzing network perturbations
Activation_statistics.m : class for collecting statistics of the HD collected over all the realizations from the neural network.
 Perturbation_statistics.m : Class for collecting Hamming distance and entropy stats
Network_act_vs_size.m  :  Simulates network activity vs. size under cross-talk/no cross-talk. Plot comparing activity in networks with/without cross-talk
Signal_perturbation_1layer.m :  Signal perturbations in 1-layer networks. Plot of mean normalized Hamming distance vs perturbation for random vs. targeted signal perturbation.
Signal_perturbation_2layer.m : Signal perturbations in 2-layer networks (with common nodes). Plots of mean normalized Hamming distance vs. perturbation.
Signal_Mutual_Information_of_HD.m :  Computes mutual information between perturbed and unperturbed layers. Plots Mutual information for multi-layered networks.
CellNumberperturbation_1layer.m :  Cell Number perturbation in 1-layer networks
Cell_Perturbation_2_layer.m : Cell Number perturbation in 2-layer networks (with common nodes)
Cell_Number_Mutual_Information_of_HD.m : Computes Mutual information between perturbed and unperturbed layers.
HD_bins_avg_values_2layer_cell_pert: Measures how perturbing nodes of varying population sizes affects network stability.
HD_bins_pop_2_layer_signal_general.m: Tests whether intrinsic node attributes (like population) affect influence across QS network topologies with different numbers of common nodes (0, 1, or 2).
HD_bins_weight_2_layer_signal_general.m : Quantify the relationship between node influence and weight strength and investigate signal perturbation robustness across different shared nodes (0, 1, or 2)
HD_vs_rankings_cell_pert.m: determine whether larger populations, stronger weights, or their combination confer greater perturbative impact. It analyzes how influence shifts in networks with 0, 1, or 2 common nodes. Generate plots that rank nodes by each feature and show corresponding changes in HD.
HD_vs_rankings.m: Quantify node influence on system behavior based on population size and sum of absolute weights. Compare trends across network configurations with 0, 1, or 2 common nodes and then generate ranked perturbation-response plots.
pop_distributions.m : Track the maximum population in each realization. Bin these maximum values across defined ranges and then visualize distribution histograms for each network configuration
Citation
If you use this codebase, please cite our upcoming paper: "Stability of Quorum Sensing Decision States in Heterogeneous Bacterial Communities".




