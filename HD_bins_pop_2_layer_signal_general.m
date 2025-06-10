% Final code for exploring importance of absolute intrinsic quanties of
% nodes in determing node influence. This code is for
% investigating the importance of Population Figures 4E. 

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
% that can be given to a network. 

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
% n_common=0 and focus only on the results of the layer on which is applied
% a perturbation


n_NN=2; % Number of Neural Networks/Layers             
net_size = 10; %the set network size (number of nodes in each layer)
num_cmn_points = 3; % the number of varying cases, in terms of num cmn nodes, that we will test 

n_samples = 1000;% the number of samples per bin
bin_ranges_pop = [0,100,200,300,400,500,600]; % ranges are  [0,100), [100,200), ...
x_bin = bin_ranges_pop + 50; % this is just for graphing purposes, so value appears in middle of a bin

pert_size = -1; %the size of the perturbation as a percentage
pert_radius=10; %the step size of the %-perturbation we are applying to a targeted node

% - - - - - - - - - - - - - S T O R E D   V A L U E S - - - - - - - - - - 

HD_avg_1_pop = zeros(num_cmn_points,length(bin_ranges_pop)); % will hold the values of avg HD for the perturbed layer
HD_avg_2_pop = zeros(num_cmn_points,length(bin_ranges_pop)); % will hold the values of avg HD for the unperturbed layer
HD_values_1_pop = zeros(num_cmn_points,n_samples,length(bin_ranges_pop)); % will hold the values of HD for the perturbed layer
HD_values_2_pop = zeros(num_cmn_points,n_samples,length(bin_ranges_pop)); % will hold the values of HD for the unperturbed layer
pop_values = zeros(num_cmn_points,n_samples,length(bin_ranges_pop)); % will hold the values of population


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - ii

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for common_node_count = 1:num_cmn_points
    n_common = common_node_count-1; % The number of common nodes in a layer
    n_pure = net_size-n_common; % The number of pure nodes in a layer
    total_num_nodes = n_NN*n_pure + n_common; %the total number of unique nodes in the entire network
    for bin=1:length(bin_ranges_pop)
        disp('Current bin:')
        disp(bin)
        for sample = 1:n_samples
            disp('Current sample: ')
            disp(sample)
            pop = -1;
            while ~(pop >= bin_ranges_pop(bin) && ((pop < bin_ranges_pop(bin) + 100) || (bin == length(bin_ranges_pop) && pop >= bin_ranges_pop(bin)))) 
                            % the last conditions takes in account the very
                            % last bin which is a special case as it can include 700 as well as 600-699
                           

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
                %    
                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     
                 % here we will perturb only nodes in layer 1
                    pert_obj_1=structural_perturbation(final_NN(1),final_NN(2),node_matrix,'AI concentration perturbation at highest cell population',1,pert_radius,time,delta_t,f);
                           
                    pert_obj_1.pert_size = pert_size;
                             
 
                 % - - - - - - - - - - - - - Calculating the HD values - - - - - - - - - - - - - %   
              
                       % getting the index of the random node that I will perturb
                       node_index = randi(net_size);
                   
                        % Perturb the random targeted node in layer 1               
                       [n_sig_HD_pert_lay,n_sig_HD_unpert_lay,pert_obj_1]=perturb_targetnode(pert_obj_1,node_index);

                       pop = NN(1).cell_pop(node_index);
               
            end
                     % calculating the resulting HD from the perturbed node                    
                     HD_values_1_pop(common_node_count,sample,bin)  = pert_obj_1.norm_hd_pertnn;                
                     HD_values_2_pop(common_node_count,sample,bin)  = pert_obj_1.norm_hd_unpert;            
                     pop_values(common_node_count,sample,bin) = pop;    
                      
       end      
    end
end


HD_values_pop = HD_values_1_pop + HD_values_2_pop; % the total Hd of both layers

% - - - -  average HD and standard deviation of HD for 0 cmn node networks - - - - % 
HD_avg_1_pop0 = mean(squeeze(HD_values_1_pop(1,:,:)));
HD_avg_2_pop0 = mean(squeeze(HD_values_2_pop(1,:,:)));
HD_avg_pop0 = mean(squeeze(HD_values_pop(1,:,:))); 

HD_err1_pop0 = std(squeeze(HD_values_1_pop(1,:,:))); 
HD_err2_pop0 = std(HD_values_2_pop(1,:,:)); 
HD_err_pop0 = std(squeeze(HD_values_pop(1,:,:))); 

% - - - -  average HD and standard deviation of HD for 1 cmn node networks - - - - % 
HD_avg_1_pop1 = mean(squeeze(HD_values_1_pop(2,:,:)));
HD_avg_2_pop1 = mean(squeeze(HD_values_2_pop(2,:,:)));
HD_avg_pop1 = mean(squeeze(HD_values_pop(2,:,:))); 

HD_err1_pop1 = std(squeeze(HD_values_1_pop(2,:,:))); 
HD_err2_pop1 = std(squeeze(HD_values_2_pop(2,:,:))); 
HD_err_pop1 = std(squeeze(HD_values_pop(2,:,:))); 

% - - - -  average HD and standard deviation of HD for 2 cmn node networks - - - - % 

HD_avg_1_pop2 = mean(squeeze(HD_values_1_pop(3,:,:)));
HD_avg_2_pop2 = mean(squeeze(HD_values_2_pop(3,:,:)));
HD_avg_pop2 = mean(squeeze(HD_values_pop(3,:,:))); 

HD_err1_pop2 = std(squeeze(HD_values_1_pop(3,:,:))); 
HD_err2_pop2 = std(HD_values_2_pop(3,:,:)); 
HD_err_pop2 = std(squeeze(HD_values_pop(3,:,:))); 

%Calling the various functions that producess the plots
HD_avg_bins(HD_avg_1_pop0,HD_avg_2_pop0,HD_avg_pop0,HD_avg_1_pop1,HD_avg_2_pop1,HD_avg_pop1,HD_avg_1_pop2,HD_avg_2_pop2,HD_avg_pop2,HD_err1_pop0,HD_err2_pop0,HD_err_pop0,HD_err1_pop1,HD_err2_pop1,HD_err_pop1,HD_err1_pop2,HD_err2_pop2,HD_err_pop2,x_bin,pert_size)   


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%
%                 V i s u a l i z a t i o n 
%
% - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -
function HD_avg_bins(HD_avg_1_pop0,HD_avg_2_pop0,HD_avg_pop0,HD_avg_1_pop1,HD_avg_2_pop1,HD_avg_pop1,HD_avg_1_pop2,HD_avg_2_pop2,HD_avg_pop2,HD_err1_pop0,HD_err2_pop0,HD_err_pop0,HD_err1_pop1,HD_err2_pop1,HD_err_pop1,HD_err1_pop2,HD_err2_pop2,HD_err_pop2,x_bin,pert_size)   

    % plotting the values for 0,1, and 2 cmn node layers to compare
    figure()
    plot(x_bin,squeeze(HD_avg_pop0(1,:,:)))
    hold on;
    plot(x_bin,squeeze(HD_avg_pop1(1,:,:)))
    hold on;
    plot(x_bin,squeeze(HD_avg_pop2(1,:,:)))
    hold on;
    legend('0 CMN Node','1 CMN Node', '2 CMN Nodes') 
    errorbar(x_bin,squeeze(HD_avg_pop0(1,:,:)),squeeze(HD_err_pop0(1,:,:)), 'LineStyle','none', 'Color','r')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_pop1(1,:,:)),squeeze(HD_err_pop1(1,:,:)), 'LineStyle','none', 'Color','r')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_pop2(1,:,:)),squeeze(HD_err_pop2(1,:,:)), 'LineStyle','none', 'Color','r')
    hold on;

    xticks([50 150 250 350 450 550 650])
    xticklabels({'0 - 99', '100-199', '200-299' , '300-399', '400-499', '500-599', '600+'})
    ylim([-0.1 .2])
    xlabel('Cell Population Range')
    ylabel('Average HD')
    title([num2str(100*pert_size), '% Signal Perturbation' ])
end

function HD_vs_pop(HD,population,pert_size)
    
    P=squeeze(population(1,:,:))
    HD1 = squeeze(HD(1,:,:))

    figure()
    scatter(squeeze(population(1,:,:)),squeeze(HD(1,:,:)))
    xlabel('Cell Population')
    ylabel('HD')
    title(['1 Common Node: ', num2str(100*pert_size), '% Signal Perturbation' ])

    figure()
    scatter(squeeze(population(2,:,:)),squeeze(HD(2,:,:)))
    xlabel('Cell Population')
    ylabel('HD')
    title(['2 Common Nodes: ', num2str(100*pert_size), '% Signal Perturbation' ])
    
end

function HD_vs_weight(HD,weights)
   
    figure()
    scatter(weights,HD)
    xlabel('Sum of Magnitude of Weights')
    ylabel('HD')

end
 
% This function institutes the time evolution of objects in the network

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


