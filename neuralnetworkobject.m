classdef neuralnetworkobject
    % This is a class which is able to create neurla network objects. These
    % objects are the representation of the threshold networks which we
    % want to analyse to understand and predict the activity of
    % heterogeneous communities of bacteria

    properties
        %n;         % no of nodes
        n_pure;     % No of pure nodes in the network
        n_common;   % Array stating No of common nodes with each network % size from 1 to n_NN-1
        n;          % It is the total of n_pure and all the n_common nodes
        weights;    % weights in the network. In our case its a bell curve at 0
        signal;     % signal concentration for each node i
        act;        % activity column for each node i
        sig_p;      % signal production rate for this network
        sig_d;      % signal decay rate for this network
        h;          % signal threshold
        avg_cells;  % average number of cell in this network
        cell_pop;   % The population distirbution in each node

        sig_time;
        
        
        %common_node_start_index; % Which of the nodes in this network is common to some other network as well
        %common_net_index; % This will store the value of the network index, with which it shares a common node

        

    end

    methods
        % Initializing our NN object
        % The maximum number of nodes in this NN is 5 for the moment
        function NN=neuralnetworkobject(n_pure,n_common,sig_p,sig_d,avg_cells)
            
            NN.n_pure=n_pure;
            NN.n_common=n_common;
            NN.n=n_pure+n_common;
            NN.weights=weight_gen(NN.n,1,0);% weight of 1 means cross-talk
           
            NN.signal=zeros(NN.n,1);
            NN.act=zeros(NN.n,1);
            NN.sig_p=sig_p;
            NN.sig_d=sig_d;
            NN.avg_cells=avg_cells;
            NN.h=(NN.avg_cells*(NN.sig_p/NN.sig_d))-1;            
            NN.sig_time=zeros(NN.n,1);
            %NN.num_common=num_comm;
            % Lets make a property for common nodes
            

            % Initializing the cell-population among the nodes
            % Then divide the total cells bewteen the nodes
            total_cells=NN.n*NN.avg_cells;

            splits = round(sort(rand(NN.n-1,1))*total_cells);
            split2 = diff([0, splits', total_cells]);
            NN.cell_pop=split2.';
        end
       
    end
end


function weights=weight_gen(n,value,mean_wt)

    switch value

        case 0 % No cross-talk takes place
            weights=eye(n,n);
    
        case 1 % Bacterial cross talk taking place
            M=normrnd(mean_wt,0.4,n);
            M=M-diag(diag(M))+eye(n,n);
            weights=M;
    end
end