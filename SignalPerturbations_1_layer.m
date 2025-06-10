% This code is written to understand all the kinds of signal perturbations
% that can be given to a network. We will be working our way with a one
% layer network first and then transition to two layer networks !

% The signal perturbations are of this form : (1) Random Node Random Signal
% Perturbation
% (2) Biggest cell population node signal perturbations


% Author : Soumya Das, Enes Haximhali
% Boedicker Lab
% University of Southern California

% This project is going to work on Perturbing a targeted node.
% I am going to perturb a target node by reducing all the signal to zero
% and do it for different realizations and see what is the mean normalized
% HD



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
n_pure=10;              % The number of pure nodes in a layer
n_common=0;             % The number of common nodes in a layer
n_realiz=3000;        % Total number of realizations
% Change en_pert
% - - - - - - - - - - - - -D e r i v e d  P a r a m e t e r s - - - - - - -
                        % Calculate the network size
net_size=n_pure+n_common;


%pert_size=-1;           % Normalized perturbation size


                        % This frequency matrix stores the stable perturbed
                        % states in each realization
%freq_mat=zeros(net_size+1,net_size+1);
% - - - - - - - - - - - E n s e m b l e .  P a r a m e t e r s - - - - - -
% -

pert_radius=10;         % How big of a perturbation do we need
en_pert=1000;            % How many realizations each perturbation is getting
n_realiz=en_pert*(2*pert_radius+1); % The total number of realizations is just
                        % the multiple of number of types of perturbations


pert_count=1;
pert_index=1;
pert_size=-pert_radius:1:pert_radius;
%pert_size=-8*ones(1,2*pert_radius+1);
pert_size=pert_size/10;




% - - - - - - - - - - - - - S T A T I S T I C S - - - - - - - - - - - - - -
pert_stat_sig=perturbation_statistics(net_size,pert_radius);
pert_stat_target_max=perturbation_statistics(net_size,pert_radius);






% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - ii

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

    pert_obj_1=structural_perturbation(final_NN(1),final_NN(2),node_matrix,'AI concentration perturbation',1,pert_radius,time,delta_t,f);

    pert_obj_2=structural_perturbation(final_NN(1),final_NN(2),node_matrix,'AI concentration perturbation at highest cell population',1,pert_radius,time,delta_t,f);

    if pert_count==en_pert
        pert_count=1;
        pert_obj_1.pert_size=pert_size(pert_index);
        pert_obj_2.pert_size=pert_size(pert_index);
        pert_index=pert_index+1;
        
    else
        pert_count=pert_count+1;
        pert_obj_1.pert_size=pert_size(pert_index);
        pert_obj_2.pert_size=pert_size(pert_index);
    end

    sig_pert_size_1=pert_obj_1.pert_size;
    sig_pert_size_2=pert_obj_2.pert_size;
 
    % Perturbing a random node in the network with a random perturbation
    [norm_sig_HD_pert_lay,norm_sig_HD_unpert_lay,pert_obj_1]=perturb_purenode(pert_obj_1);

    % Perturb the maximum population nodes signal vector with random
    % perturbation
    pop=pert_obj_1.pert_nn.cell_pop;
    [xx,perturb_node]=max(pop);
    [n_sig_HD_pert_lay,n_sig_HD_unpert_lay,pert_obj_2]=perturb_targetnode(pert_obj_2,perturb_node);

    %norm_sig_HD_pert_lay=sig_HD_pert_lay/net_size;
    pert_stat_sig=collecting_data(pert_stat_sig,norm_sig_HD_pert_lay,sig_pert_size_1);
    pert_stat_target_max=collecting_data(pert_stat_target_max,n_sig_HD_pert_lay,sig_pert_size_2);

    

    %Here we assign the normalized hd in the perturb ation statistics object
end
f1=figure(1);
hold on;
pert_stat_sig.plotter("AI Concentration Perturbations",'b'); 
pert_stat_target_max.plotter("AI Concentration Perturbations",'r');

hold off;
legend({'Random Perturbation','Targeted Perturbation'});


% - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - -
%       This is plotting the signal concentration vs time 
%       for all the times before and after the perturbations
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%f2=figure(2);
%for i=1:net_size
%    yline(pert_obj_1.final_pert_nn.h);
%    plot(pert_obj_1.final_pert_nn.sig_time(i,1:2*time));
%    hold on;
%end
%hold off;





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
    


function [NN,logic_count]= shared_nodes_setting(NN,node_matrix)%logic_count)
            
            
    for i=1:length(NN)-1 %n_NN is the number of neural networks in the macro-system
        for j=1:length(node_matrix)
            if i~=j 
                x=NN(i).n_pure;
                y=x+node_matrix(j);
                x1=1;
                y1=NN(j).n_pure;
                %NN(i).cell_pop=cat(1,NN(i).cell_pop,split2.');
                NN(j).cell_pop=cat(1,NN(j).cell_pop(x1:y1,1),NN(i).cell_pop(x+1:y,1));
               
            end
        end
    end
end

