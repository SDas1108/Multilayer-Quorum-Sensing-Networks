% This code is giving me the mean HD vs perturbation size for different
% number of common nodes. We will stick from 0,1,2 common nodes

% This code is written to understand all the kinds of cell-number perturbations
% that can be given to a network. We will be working our way with a one
% layer network first and then transition to two layer networks !

% The signal perturbations are of this form : (1) Random Node Cell Number
% Perturbations and 
% (2) Biggest cell population node cell number perturbations


% This project is going to work on Perturbing a targeted node.
% I am going to perturb a target node by reducing all the signal to zero
% and do it for different realizations and see what is the mean normalized
% HD


% Author : Soumya Das, Enes Haximhali
% Boedicker Lab
% University of Southern California



% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - i
   
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

tic
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
%n_realiz=1000;          % Total number of realizations

% - - - - - - - - - - - - -D e r i v e d  P a r a m e t e r s - - - - - - -
                        % Calculate the network size
net_size=n_pure+n_common;

pert_radius=10;         % How big of a perturbation do we need
                        % Normalized perturbation size


% - - - - - - - - - - - E n s e m b l e .  P a r a m e t e r s - - - - - -
% -
en_pert=2000;            % How many realizations each perturbation is getting
%n_realiz=1;
n_realiz=en_pert*(2*pert_radius+1); % The total number of realizations is just
                        % the multiple of number of types of perturbations
pert_count=1;
pert_index=1;
pert_size=-pert_radius:1:pert_radius;
pert_size=pert_size/10;


                        % This frequency matrix stores the stable perturbed
                        % states in each realization
%freq_mat=zeros(net_size+1,net_size+1);

% - - - - - - - - - - - - - R A T I O S - - - - - - - - - - - - - - - - - -

%das_ratio=zeros(1,net_size+1);


% - - - - - - - - - - - - - S T A T I S T I C S - - - - - - - - - - - - - -
pert_stat_pert=perturbation_statistics(net_size,pert_radius);
pert_stat_unpert=perturbation_statistics(net_size,pert_radius);
%pert_stat_target_max_cn=perturbation_statistics(pert_radius);

color=['r','k','b','c','m','y'];
g=1;
% - - - - - - - - - - - - - - - - ----------- - - - -------- - - ----------



% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - ii

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for n_pure= net_size-1:-1:net_size-2
    n_common=net_size-n_pure;
    pert_count=1;
    pert_index=1;
    pert_stat_pert=perturbation_statistics(net_size,pert_radius);
    pert_stat_unpert=perturbation_statistics(net_size,pert_radius);
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
    %
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
    % I need to make sure that the net_signal is correct in each case

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

    pert_obj_1=structural_perturbation(final_NN(1),final_NN(2),node_matrix,'Common Node Random Signal Perturbation',1,pert_radius,time,delta_t,f);
    %pert_obj_2=structural_perturbation(final_NN(1),final_NN(2),node_matrix,'Random Perturbation at highest cell population',2,pert_radius,time,delta_t,f);

    if pert_count==en_pert
        pert_count=1;
        pert_obj_1.pert_size=pert_size(pert_index);
       % pert_obj_2.pert_size=pert_size(pert_index);
        pert_index=pert_index+1;
        
    else
        pert_count=pert_count+1;
        pert_obj_1.pert_size=pert_size(pert_index);
        pert_obj_2.pert_size=pert_size(pert_index);
    end
    
    cn_pert_size_1=pert_obj_1.pert_size;

 
    % Perturbing a random node in the network with a random perturbation
    [nhd_p,nhd_up,pert_obj_1]=perturb_commonnode(pert_obj_1);
 
    
    % The way to get the unperturbed layer HD vs perturbation graph is to 
    % change nhd_p to nhd_up which is normalized hamming distance of
    % perturbed layer to unperturbed layer
    pert_stat_pert=collecting_data(pert_stat_pert,nhd_p,cn_pert_size_1);

    pert_stat_unpert=collecting_data(pert_stat_unpert,nhd_up,cn_pert_size_1);
 


    
end
pert_layer(n_common+1).stat=pert_stat_pert.pert_data;
unpert_layer(n_common+1).stat=pert_stat_unpert.pert_data;

f1=figure(1);
hold on;
pert_stat_pert.plotter("Perturbed Signal Perturbations",color(g));
ylim([-0.05 0.1]);
f2=figure(2);
hold on;
pert_stat_unpert.plotter("Unperturbed Signal Perturbations",color(g)); 
ylim([-0.05 0.1]);
g=g+1;

end
hold off;
legend('1','2');

toc



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
    

function [NN]=shared_nodes_setting(NN,node_matrix)

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

%function [NN,logic_count]= shared_nodes_setting(NN,node_matrix)%logic_count)
            
            
%    for i=1:length(NN)-1 %n_NN is the number of neural networks in the macro-system
%        for j=1:length(node_matrix)
%            if i~=j 
%                x=NN(i).n_pure;
%                y=x+node_matrix(j);
%                x1=1;
%               y1=NN(j).n_pure;
                %NN(i).cell_pop=cat(1,NN(i).cell_pop,split2.');
%                NN(j).cell_pop=cat(1,NN(j).cell_pop(x1:y1,1),NN(i).cell_pop(x+1:y,1));
               
%            end
%        end
%    end
%end

