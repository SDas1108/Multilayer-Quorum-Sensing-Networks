% Plots network activity as a function of network size



\

% Author : Soumya Das, Enes Haximhali
% Boedicker Lab
% University of Southern California




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

n_NN=2;                 % Number of Neural Networks/Layers
n_common=0;             % No common nodes
max_n_pure=20;              % Here it means the maximum number of Pure nodes we want in a network
n_realiz=100;        % Total number of realizations

weight_type=[0 1];          % 0= No Cross-Talk
                             % 1= Cross-Talk

% - - - - - - - - - - - - -D e r i v e d  P a r a m e t e r s - - - - - - -

% Creating two activation_statistics object which will help me in finding
% the activity of the neural network

% Here n_pure is the maximum number of pure nodes we want to give to the
% network !
for i=1:n_NN
    network_stat(i).stat=activation_statistics(0,0,max_n_pure,max_n_pure,n_common,0,n_realiz);
end






% - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - - - 

%                       S E C T I O N - ii

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
for n_pure=2:20

    net_size=n_pure;
for realizations=1:n_realiz

    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    % Creating 2 Neural Network Object - One is cross-talk
    % The other is no cross-talk
    
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    diagonal = [n_pure,n_pure];% we are going to get a random number of pure node in each realization
    t=zeros(n_NN,n_NN);
    t(1,n_NN)=n_common;
    node_matrix = diag(diagonal)+t+t.';
    
    for i=1:2
        NN(i)=neuralnetworkobject(n_pure,0,p,d,avg_cells,weight_type(i));
    end
    
    
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    %
    %   Simulating the neural network through a time series evolution. Here
    %   the network evolves in its signal concentration and activity states
    %   with respect to a set amount of time. The final activity is
    %   measured to understand our multilayer network !
    % 
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    [final_NN]=time_series(NN,n_NN,node_matrix,time,delta_t,f);

    for j=1:n_NN
        network_stat(j).stat=collectingdata(network_stat(j).stat,final_NN(j).signal,final_NN(j).act,final_NN(j).n,final_NN(j).n_common,final_NN(j).h,realizations);
    end
   
end
end
% This one plots activity as network size
network_stat(1).stat.plotter(network_stat(2).stat,4);



function [final_NN,net_sig] = time_series(NN,n_NN,node_matrix,time,delta_t,f)
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
    

