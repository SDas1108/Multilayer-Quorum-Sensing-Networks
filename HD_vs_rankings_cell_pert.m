
% Final code for finding the activity and the sensitivity of the QS
% networks 

% Author : Soumya Das, Enes Haximhali
% Boedicker Lab
% University of Southern California

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - i

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


% - - - - - - - - - - -  I n p u t   P a r a m e t e r s - - - - - - - - - 
clc;
clear all;
p=1.5;                   % Signal Production rate of both layers
d=1;                     % Signal Degradation rate of % This code is written to understand all the kinds of signal perturbations
% that can be given to a network. We will be working our way with a one
% layer network first and then transition to two layer networks !

% The signal perturbations are of this form : (1) Random Node Random Signal
% Perturbation
% (2) Biggest cell population node signal perturbations

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - i
   
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


% - - - - - - - - - - -  I n p u t   p a r a m e t e r s - - - - - - - - - 
clc;
clear all;
p=1.5;                   % Signal Production rate of both layers
d=1;                     % Signal Degradation rate of both layers
avg_cells=100;           % Average number of cells in the model
time=200;                % Simulation time in each realization
delta_t=0.1;             % Value of each time step
f=4;                     % Fold Change 

% - - - - - - - - - - - I n p u t   L a y e r  P a r a  - - - - - - - - - 
% If we need to work for a 1 layer network we set the n_NN=2 and set the 
% n_common=0


n_NN=2;                 % Number of Neural Networks/Layers             
n_realiz=10000;          % Total number of realizations
net_size = 10; %the set network size (number of nodes in each layer)

pert_size = -.9; %the size of the perturbation as a percentage
pert_radius=10; %the step size of the %-perturbation we are applying to a targeted node


num_cmn_points = 3;  %the total number of networks with different, incrementing, number of common nodes, starting with 0



% - - - - - - - - - - - - - S T O R E D   V A L U E S - - - - - - - - - - 
HD_values_1 = zeros(n_realiz,net_size,num_cmn_points);
HD_values_2 = zeros(n_realiz,net_size,num_cmn_points);


Ranking = NaN(net_size,num_cmn_points); %will hold the values for ordered ranking of HD_values based on pop size, weights, etc from 1
Ranking(1,:) = 1;

Population_values = zeros(n_realiz,net_size,num_cmn_points); %will store the population values of all nodes perturbed
Weights_sum_values = zeros(n_realiz,net_size,num_cmn_points); %will store the sum of weights connected to perturbed node
weighted_pop_values = zeros(n_realiz,net_size,num_cmn_points); %will store the log of population*sum of weights connected to perturbed node


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - ii

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for common_node_count = 1:num_cmn_points
    n_common = common_node_count- 1; % The number of common nodes in a layer
    n_pure = net_size-n_common; % The number of pure nodes in a layer
    total_num_nodes = n_NN*n_pure + n_common; %the total number of unique nodes in the entire network

    %filling the rank values from 1 to n_pure
    for p = 2:net_size       
        Ranking(p,common_node_count) = Ranking(p-1,common_node_count)+1;
    end

    for realizations=1:n_realiz
    
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Creating a node matrix to tell us about the number of pure and common
        % nodes in both the layers.
        % Node matrix is a symmetric matrix telling me that the diagonal 
        % elements are the pure nodes and the off diagonal are the shared nodes.
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        diagonal = [n_pure,n_pure];
        t=zeros(n_NN,n_NN);
        t(1,n_NN)=n_common;
        node_matrix = diag(diagonal)+t+t.';
        
                  
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        % Creating 2 neural network objects. This neural network object holds
        % important properties : 
        %               (i) weights of the layer
        %               (ii)cell population vector for the population at each
        %               node.
        %               (iii)Signal vector for signal concentration at each
        %               node.
        %               (iv) Activity vector for letting us know if any node is
        %               active = 1 or inactive = 0
        % 
        % Once the neual network object is created we call the
        % shared_nodes_setting() fn to make sure both the neural network are
        % sharing the same population of cells at the common node.
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        c=2;
        for i=1:n_NN
            NN(i)=neuralnetworkobject(node_matrix(i,i),node_matrix(i,c), ...
                p,d,avg_cells);
            c=1;
        end
        NN=shared_nodes_setting(NN,node_matrix);  
         
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        %
        %   Simulating the neural network through a time series evolution. Here
        %   the network evolves in its signal concentration and activity states
        %   with respect to a set amount of time. The final activity is
        %   measured to understand our multilayer network !
        % 
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        final_NN=time_series(NN,n_NN,node_matrix,time,delta_t,f);
    
      
        
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        %
        %
        %   Creating a perturbation object which can handle perturbing the two
        %   networks and then sending us the normalized hamming distance as an
        %   output of that perturbation.
        %
        %   Remember we had already taken the following inputs for the rest of
        %   the section to work : 
        %                       (i) Perturabtion size
        %                       (ii) Perturbation target
    
        %   About the perturbation : Remember in this simulation we are 
        %   targeting the highest population node and decreasing its signal 
        %   concentration to 0 in one time step temporarily
        %   
        %    
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             
         %here we will perturb only nodes in layer 1
            pert_obj_1=structural_perturbation(final_NN(1),final_NN(2),node_matrix,'AI concentration perturbation at highest cell population',2,pert_radius,time,delta_t,f);
                   
            pert_obj_1.pert_size = pert_size;
                     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % - - - - - - - - - - - - - L I N E A R   R E S P O N S E   M A T R I X - - - - - - - - - - - - - %   
            for node_index = 1:net_size   
                % Perturb the current targeted node with random perturbation                    
                 [n_sig_HD_pert_lay,n_sig_HD_unpert_lay,pert_obj_1]=perturb_targetnode(pert_obj_1,node_index);
          
                 % calculating HD values
                 HD_values_1(realizations,node_index,common_node_count) = pert_obj_1.norm_hd_pertnn;
                 
                 HD_values_2(realizations,node_index,common_node_count) = pert_obj_1.norm_hd_unpert;
                 
                % calculating population and weight sum values values
                 Population_values(realizations,node_index,common_node_count) = NN(1).cell_pop(node_index);                 
                 Weights_sum_values(realizations,node_index,common_node_count) = sum(NN(1).weights(:,node_index));

                 weighted_pop_values(realizations,node_index,common_node_count) = log10(NN(1).cell_pop(node_index)*sum(NN(1).weights(:,node_index)));
                 %weighted_pop_values(realizations,node_index,common_node_count) = sum(abs(NN(1).weights(:,node_index)))+log10(NN(1).cell_pop(node_index));
            end        
    end
end

%Now create a copy of reordered HD_values by population, weights, etc

HD_values_1_ord_pop = HD_values_1;
HD_values_1_ord_weight = HD_values_1;
HD_values_1_ord_wp = HD_values_1; % for the log of the product*sum_weight

HD_values_2_ord_pop = HD_values_2;
HD_values_2_ord_weight = HD_values_2; 
HD_values_2_ord_wp = HD_values_2;

%first sort the x values and then sort the y values in the same order

pop_values = NaN(net_size,1); %will temporalily hold the ordered population values
weight_values = NaN(net_size,1);  %will temporalily hold the ordered weight values
weighted_pop = NaN(net_size,1);  %will temporalily hold the ordered log of pop*weight_sum values

for i = 1:num_cmn_points
    for j = 1:n_realiz

        %the current set of n_pure HD_values that we want to organize
       
        current_HD1 = HD_values_1(j,:,i);
        current_HD2 = HD_values_2(j,:,i);
        
        %sorting the HD_values
        [pop_values,I] = sort(Population_values(j,:,i));    
        
        ordered_HD1_pop = current_HD1(I);
        ordered_HD2_pop = current_HD2(I);

    
        [weight_values,I2] = sort(Weights_sum_values(j,:,i));    
         
        ordered_HD1_weight = current_HD1(I2);
        ordered_HD2_weight = current_HD2(I2);


        [weighted_pop,I3] = sort(Weights_sum_values(j,:,i));    
         
        ordered_HD1_wp = current_HD1(I3);
        ordered_HD2_wp = current_HD2(I3);


        HD_values_1_ord_pop(j,:,i) = ordered_HD1_pop;
        HD_values_1_ord_weight(j,:,i) = ordered_HD1_weight;
        HD_values_1_ord_wp(j,:,i) = ordered_HD1_wp;

        HD_values_2_ord_pop(j,:,i) = ordered_HD2_pop;
        HD_values_2_ord_weight(j,:,i) = ordered_HD2_weight;
        HD_values_2_ord_wp(j,:,i) = ordered_HD2_wp;

    end
end
        
HD_values_tot_ord_pop = HD_values_1_ord_pop + HD_values_2_ord_pop;
HD_values_tot_ord_weight = HD_values_1_ord_weight + HD_values_2_ord_weight;
HD_values_tot_ord_wp = HD_values_1_ord_wp + HD_values_2_ord_wp;

%Calling the various functions that producess the visualizations


HD1_vs_Pop(Ranking,mean(HD_values_1_ord_pop(:,:,1),"omitnan"),mean(HD_values_1_ord_pop(:,:,2),"omitnan"),mean(HD_values_1_ord_pop(:,:,3),"omitnan"),std(HD_values_1_ord_pop(:,:,1),"omitnan"),std(HD_values_1_ord_pop(:,:,2),"omitnan"),std(HD_values_1_ord_pop(:,:,3),"omitnan"),pert_size);

HD2_vs_Pop(Ranking,mean(HD_values_2_ord_pop(:,:,1),"omitnan"),mean(HD_values_2_ord_pop(:,:,2),"omitnan"),mean(HD_values_2_ord_pop(:,:,3),"omitnan"),std(HD_values_2_ord_pop(:,:,1),"omitnan"),std(HD_values_2_ord_pop(:,:,2),"omitnan"),std(HD_values_2_ord_pop(:,:,3),"omitnan"),pert_size);%

HDtot_vs_Pop(Ranking,mean(HD_values_tot_ord_pop(:,:,1),"omitnan"),mean(HD_values_tot_ord_pop(:,:,2),"omitnan"),mean(HD_values_tot_ord_pop(:,:,3),"omitnan"),std(HD_values_tot_ord_pop(:,:,1),"omitnan"),std(HD_values_tot_ord_pop(:,:,2),"omitnan"),std(HD_values_tot_ord_pop(:,:,3),"omitnan"),pert_size);


HD1_vs_weight(Ranking,mean(HD_values_1_ord_weight(:,:,1),"omitnan"),mean(HD_values_1_ord_weight(:,:,2),"omitnan"),mean(HD_values_1_ord_weight(:,:,3),"omitnan"),std(HD_values_1_ord_weight(:,:,1),"omitnan"),std(HD_values_1_ord_weight(:,:,2),"omitnan"),std(HD_values_1_ord_weight(:,:,3),"omitnan"),pert_size);

HD2_vs_weight(Ranking,mean(HD_values_2_ord_weight(:,:,1),"omitnan"),mean(HD_values_2_ord_weight(:,:,2),"omitnan"),mean(HD_values_2_ord_weight(:,:,3),"omitnan"),std(HD_values_2_ord_weight(:,:,1),"omitnan"),std(HD_values_2_ord_weight(:,:,2),"omitnan"),std(HD_values_2_ord_weight(:,:,3),"omitnan"),pert_size);

HDtot_vs_weight(Ranking,mean(HD_values_tot_ord_weight(:,:,1),"omitnan"),mean(HD_values_tot_ord_weight(:,:,2),"omitnan"),mean(HD_values_tot_ord_weight(:,:,3),"omitnan"),std(HD_values_tot_ord_weight(:,:,1),"omitnan"),std(HD_values_tot_ord_weight(:,:,2),"omitnan"),std(HD_values_tot_ord_weight(:,:,3),"omitnan"),pert_size);


HD1_vs_wp(Ranking,mean(HD_values_1_ord_wp(:,:,1),"omitnan"),mean(HD_values_1_ord_wp(:,:,2),"omitnan"),mean(HD_values_1_ord_wp(:,:,3),"omitnan"),std(HD_values_1_ord_wp(:,:,1),"omitnan"),std(HD_values_1_ord_wp(:,:,2),"omitnan"),std(HD_values_1_ord_wp(:,:,3),"omitnan"),pert_size);

HD2_vs_wp(Ranking,mean(HD_values_2_ord_wp(:,:,1),"omitnan"),mean(HD_values_2_ord_wp(:,:,2),"omitnan"),mean(HD_values_2_ord_wp(:,:,3),"omitnan"),std(HD_values_2_ord_wp(:,:,1),"omitnan"),std(HD_values_2_ord_wp(:,:,2),"omitnan"),std(HD_values_2_ord_wp(:,:,3),"omitnan"),pert_size);

HDtot_vs_wp(Ranking,mean(HD_values_tot_ord_wp(:,:,1),"omitnan"),mean(HD_values_tot_ord_wp(:,:,2),"omitnan"),mean(HD_values_tot_ord_wp(:,:,3),"omitnan"),std(HD_values_tot_ord_wp(:,:,1),"omitnan"),std(HD_values_tot_ord_wp(:,:,2),"omitnan"),std(HD_values_tot_ord_wp(:,:,3),"omitnan"),pert_size);


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - -
%
%                 V i s u a l i z a t i o n 
%
% - - - - - - - - - - - - - - - - - - - - - - - -- - - -  - - - - - - -

function HD1_vs_Pop(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('HD of Perturbed Layer')
    xlabel('Population Size Ranking - Least to Greatest')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

function HD2_vs_Pop(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('HD of Unperturbed Layer')
    xlabel('Population Size Ranking - Least to Greatest')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

function HDtot_vs_Pop(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('Sum of HD in Both Layers')
    xlabel('Population Size Ranking - Least to Greatest')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

%- - - - - - - - -  -- - -

function HD1_vs_weight(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('HD of Perturbed Layer')
    xlabel('Weight Sum Ranking - Least to Greatest')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

function HD2_vs_weight(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('HD of Unperturbed Layer')
    xlabel('Weight Sum Ranking - Least to Greatest')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

function HDtot_vs_weight(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('Sum of HD in Both Layers')
    xlabel('Weight Sum Ranking - Least to Greatest')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end


% - - - - - - -  - - - - -  - -%
function HD1_vs_wp(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('HD of Perturbed Layer')
    xlabel('log(Population*Sum of Weights)')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

function HD2_vs_wp(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('HD of Unperturbed Layer')
    xlabel('log(Population*Sum of Weights)')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

function HDtot_vs_wp(Ranking_pure,Z_0CMN,Z_1CMN,Z_2CMN,err_0CMN,err_1CMN,err_2CMN,pert_size)
    figure();
    scatter(Ranking_pure(:,1),Z_0CMN);
    hold on;
    scatter(Ranking_pure(:,2),Z_1CMN);
    hold on;
    scatter(Ranking_pure(:,3),Z_2CMN);
    hold on;

    legend('0 CMN Nodes','1 CMN Nodes','2 CMN Nodes')
    
    errorbar(Ranking_pure(:,1),Z_0CMN,err_0CMN,'b');
    hold on;
    errorbar(Ranking_pure(:,2),Z_1CMN,err_1CMN,'r');
    hold on;
    errorbar(Ranking_pure(:,3),Z_2CMN,err_2CMN,'y');
    
    ylabel('Sum of HD in Both Layers')
    xlabel('log(Population*Sum of Weights)')
    title(['Cell Perturbation Size: ' num2str(pert_size*100) '%'])
end

function [mean_y_values,error_y_values,quantized_x_values,ordered_y_values,ordered_x_values,x_uncertainty] = find_mean_values(y_values,x_values,num_points)

    %first sort the x values and then sort the y values in the same order
    [ordered_x_values,I] = sort(x_values);

    ordered_y_values = y_values(I); 

    %the individual quantixation of the x_axis
    quantization = (ordered_x_values(end)-ordered_x_values(1))/num_points;
    x_uncertainty = quantization/2;
  
    quantized_x_values = (ordered_x_values(1)+quantization/2):quantization:(ordered_x_values(end)-quantization/2);
    quantized_x_values = quantized_x_values.';

    mean_y_values = zeros(length(quantized_x_values),1);
    error_y_values = zeros(length(quantized_x_values),1);%zeros(length(quantized_x_values),1);
    array_counter = 1; %used to keep track of the current index of the ordered_x_values that are looking at
   
   
   check = 0;
   for i = 1:(length(quantized_x_values)-1)
      group_counter = 1; %used to keep track of how many elements in a single group
      group = [];
      check = check + 1;
      disp(check);
      
        while ((ordered_x_values(array_counter) >= (quantized_x_values(i)-quantization/2)) && (ordered_x_values(array_counter) < (quantized_x_values(i)+quantization/2)))
            group(group_counter) = ordered_y_values(array_counter);
            group_counter = group_counter + 1;
            array_counter = array_counter + 1;          
              
        end 
          
        mean_y_values(i) = mean(group,'omitnan');
        error_y_values(i) = std(group,'omitnan');
     
        
   end

  %Here we delete the NaN values(a NaN value means that no point was found in the matrix)
   NaN_loc = find(isnan(mean_y_values)); %first finding the indices of the NaN values

   mean_y_values(NaN_loc) = [];
   error_y_values(NaN_loc) = [];
   quantized_x_values(NaN_loc) = [];

end





function [final_NN,net] = time_series(NN,n_NN,node_matrix,time,delta_t,f)
    % net is a structure that is holding the signal evolution of both the
    % networks 
    %This object function is performing the first time evolution of
    %the system and returning the final activity state as a vector
    
  
    
    for t=1:time
        for i=1:n_NN
           
            factor_1=NN(i).sig_p*delta_t*(ones(NN(i).n,1)+(f-1)*NN(i).act);
            factor_2=1-NN(i).sig_d*delta_t;
            NN(i).signal=factor_2*NN(i).signal+factor_1.*NN(i).cell_pop;
            NN(i).sig_time=cat(2,NN(i).sig_time,NN(i).signal);
            
            
            total_input_signal=NN(i).weights.'*NN(i).signal;
            response=total_input_signal-NN(i).h;
            NN(i).act=heaviside(response);
            pert_id=find(NN(i).cell_pop==0);
            if pert_id
                NN(i).act(pert_id)=0;
            end
        end
        
        if NN(1).n_common>0
        
    
            for i=1:length(NN)-1
                for j=1:length(NN)
                    if i~=j
                        x=NN(i).n_pure;                    
                        y=x+node_matrix(i,j);
                        x1=NN(j).n_pure;
                        y1=x1+node_matrix(j,i);
                        NN(i).act(x+1:y)=NN(i).act(x+1:y)| NN(j).act(x1+1:y1);
                        NN(j).act(x1+1:y1)=NN(i).act(x+1:y);
                    end
                end
            end
        end
    final_NN=NN;
    end
end
    


function [NN,logic_count]= shared_nodes_setting(NN,node_matrix)

    C=node_matrix(1,2);
    P=node_matrix(1,1);
    total_net_size=2*P+C;
    total_cells=NN(1).avg_cells*(total_net_size);
    splits = round(sort(rand(total_net_size-1,1)*total_cells));
    split2 = diff([0, splits', total_cells]);
    full_pop=split2.';
    % First filling the pure nodes
    NN(1).cell_pop(1:P,1)=full_pop(1:P,1);
    NN(2).cell_pop(1:P,1)=full_pop(P+1:2*P,1);
    % Now filling the common nodes
    NN(1).cell_pop(P+1:P+C,1)=full_pop(2*P+1:total_net_size,1);
    NN(2).cell_pop(P+1:P+C,1)=full_pop(2*P+1:total_net_size,1);
end


