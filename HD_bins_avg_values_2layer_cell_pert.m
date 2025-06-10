
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
n_realiz=1;          % Total number of realizations
net_size = 10; %the set network size (number of nodes in each layer)
num_cmn_points = 2; 

n_samples = 100;% the number of samples per bin
bin_ranges_pop = [0,100,200,300,400,500,600]; % ranges are  [0,100), [100,200), ...
x_bin = bin_ranges_pop + 50; % this is just for graphing purposes, so value appears in middle of a bin

pert_size = -.9; %the size of the perturbation as a percentage
pert_radius=10; %the step size of the %-perturbation we are applying to a targeted node

% - - - - - - - - - - - - - S T O R E D   V A L U E S - - - - - - - - - - 

HD_avg_1_pop = zeros(num_cmn_points,length(bin_ranges_pop)); % will hold the values of HD for the perturbed layer
HD_avg_2_pop = zeros(num_cmn_points,length(bin_ranges_pop)); % will hold the values of HD for the unperturbed layer
HD_values_1_pop = zeros(num_cmn_points,n_samples,length(bin_ranges_pop));
HD_values_2_pop = zeros(num_cmn_points,n_samples,length(bin_ranges_pop)); 
Population_values = zeros(num_cmn_points,n_samples,length(bin_ranges_pop)); 


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - ii

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for common_node_count = 1:num_cmn_points
    n_common = common_node_count; % The number of common nodes in a layer
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
            
                %   About the perturbation : Remember in this simulation we are 
                %   targeting the highest population node and decreasing its signal 
                %   concentration to 0 in one time step temporarily
                %   
                %    
                % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     
                 % here we will perturb only nodes in layer 1
                    pert_obj_1=structural_perturbation(final_NN(1),final_NN(2),node_matrix,'AI concentration perturbation at highest cell population',2,pert_radius,time,delta_t,f);
                           
                    pert_obj_1.pert_size = pert_size;
                             
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                     Population_values(common_node_count,sample,bin) = pop;    
                      
       end      
    end
end

HD_values_pop = HD_values_1_pop + HD_values_2_pop;

HD_avg_1_pop1 = mean(squeeze(HD_values_1_pop(1,:,:)));
HD_avg_2_pop1 = mean(squeeze(HD_values_2_pop(1,:,:)));
HD_avg_pop1 = mean(squeeze(HD_values_pop(1,:,:))); 

HD_avg_1_pop2 = mean(squeeze(HD_values_1_pop(2,:,:)));
HD_avg_2_pop2 = mean(squeeze(HD_values_2_pop(2,:,:)));
HD_avg_pop2 = mean(squeeze(HD_values_pop(2,:,:))); 

HD_err1_pop1 = std(squeeze(HD_values_1_pop(1,:,:))); 
HD_err2_pop1 = std(squeeze(HD_values_2_pop(1,:,:))); 
HD_err_pop1 = std(squeeze(HD_values_pop(1,:,:))); 

HD_err1_pop2 = std(squeeze(HD_values_1_pop(1,:,:))); 
HD_err2_pop2 = std(HD_values_2_pop(1,:,:)); 
HD_err_pop2 = std(squeeze(HD_values_pop(1,:,:))); 

%Calling the various functions that producess the visualizations
HD_avg_bins(HD_avg_1_pop1,HD_avg_2_pop1,HD_avg_pop1,HD_avg_1_pop2,HD_avg_2_pop2,HD_avg_pop2,HD_err1_pop1,HD_err2_pop1,HD_err_pop1,HD_err1_pop2,HD_err2_pop2,HD_err_pop2,x_bin,pert_size)   
HD_vs_pop(HD_values_pop,Population_values,pert_size) 

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%
%                 V i s u a l i z a t i o n 
%
% - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - -
function HD_avg_bins(HD_avg_1_pop1,HD_avg_2_pop1,HD_avg_pop1,HD_avg_1_pop2,HD_avg_2_pop2,HD_avg_pop2,HD_err1_pop1,HD_err2_pop1,HD_err_pop1,HD_err1_pop2,HD_err2_pop2,HD_err_pop2,x_bin,pert_size)   
    
    % plotting the values for 1 cmn node layer
    figure()
    plot(x_bin,squeeze(HD_avg_1_pop1(1,:,:)),'Color','b')
    hold on;
    plot(x_bin,squeeze(HD_avg_2_pop1(1,:,:)),'Color','r')
    hold on;
    plot(x_bin,squeeze(HD_avg_pop1(1,:,:)),'Color','y')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_1_pop1(1,:,:)),squeeze(HD_err1_pop1(1,:,:)), 'LineStyle','none', 'Color','b')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_2_pop1(1,:,:)),squeeze(HD_err2_pop1(1,:,:)), 'LineStyle','none', 'Color','r')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_pop1(1,:,:)),squeeze(HD_err_pop1(1,:,:)), 'LineStyle','none', 'Color','y')
    hold on;

    legend('Perturbed Layer', 'Unperturbed Layer', 'Both Layers')
    title(['1 Common Node: ', num2str(100*pert_size), '% Cell Perturbation' ])
    xticks([50 150 250 350 450 550 650])
    xticklabels({'0 - 99', '100-199', '200-299' , '300-399', '400-499', '500-599', '600+'})
    xlabel('Cell Population Range')
    ylabel('Average HD')

    % plotting the values for 2 cmn node layer
    figure()
    plot(x_bin,squeeze(HD_avg_1_pop2(1,:,:)),'Color','b')
    hold on;
    plot(x_bin,squeeze(HD_avg_2_pop2(1,:,:)),'Color','r')
    hold on; 
    plot(x_bin,squeeze(HD_avg_pop2(1,:,:)),'Color','y')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_1_pop2(1,:,:)),squeeze(HD_err1_pop2(1,:,:)), 'LineStyle','none', 'Color','b')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_2_pop2(1,:,:)),squeeze(HD_err2_pop2(1,:,:)), 'LineStyle','none', 'Color','r')
    hold on;
    errorbar(x_bin,squeeze(HD_avg_pop2(1,:,:)),squeeze(HD_err_pop2(1,:,:)), 'LineStyle','none', 'Color','y')
    hold on;
    
    legend('Perturbed Layer', 'Unperturbed Layer', 'Both Layers')
    title(['2 Common Node: ', num2str(100*pert_size), '% Cell Perturbation' ])
    xticks([50 150 250 350 450 550 650])
    xticklabels({'0 - 99', '100-199', '200-299' , '300-399', '400-499', '500-599', '600+'})
    xlabel('Cell Population Range')
    ylabel('Average HD')

    % plotting the values for 1 and 2 cmn node layers to compare
    figure()
    plot(x_bin,squeeze(HD_avg_pop1(1,:,:)))
    hold on;
    plot(x_bin,squeeze(HD_avg_pop2(1,:,:)))
    legend('1 CMN Node', '2 CMN Nodes') 
    xticks([50 150 250 350 450 550 650])
    xticklabels({'0 - 99', '100-199', '200-299' , '300-399', '400-499', '500-599', '600+'})
    xlabel('Cell Population Range')
    ylabel('Average HD')
    title([num2str(100*pert_size), '% Cell Perturbation' ])
end

function HD_vs_pop(HD,population,pert_size)

    
    P=squeeze(population(1,:,:))
    HD1 = squeeze(HD(1,:,:))

    figure()
    scatter(squeeze(population(1,:,:)),squeeze(HD(1,:,:)))
    xlabel('Cell Population')
    ylabel('HD')
    title(['1 Common Node: ', num2str(100*pert_size), '% Cell Perturbation' ])

    figure()
    scatter(squeeze(population(2,:,:)),squeeze(HD(2,:,:)))
    xlabel('Cell Population')
    ylabel('HD')
    title(['2 Common Node: ', num2str(100*pert_size), '% Cell Perturbation' ])
    
end

function HD_vs_weight(HD,weights)
   
    figure()
    scatter(weights,HD)
    xlabel('Sum of Magnitude of Weights')
    ylabel('HD')

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
    error_y_values = zeros(length(quantized_x_values),1);
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


