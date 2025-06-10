classdef activation_statistics
  % These are the statistics class made to measure the statistics of the
  % data collected over all the realizations from the neural network
  % objects created. It then plots the different measure quantities like
  % Mean activity as a function of network size, no of common nodes, signal concentration vs time
    properties
        mean_active;
        n_pure;
        n_common;
        thresh;
        % The signal over time measurements. This going to store the signal
        % for all the nodes
        signal_time;
        ma_network_size;   % To measure the mean fracton of active nodes as a function of network size
        counter_size;
        ma_size_r;
        std_size;
        ma_matrix;          % This is a max_node x No of realizations matrix storing 
                            % the mean activity in each realziation for each node
        
        ma_common;   % To measure the mean fracton of active nodes as a function of common nodes
        counter_common;
        ma_common_r;
        ma_common_mat;
        std_common;
        flag=0; % This will tell if ma_commons first position is the 0 common node or not !


        %act_data=struct('ma_matrix',[0],'ma_net_size',[0],'ma_net_std',[0],'ma_common',[0],'ma_common_std','signal_time_series',[0]);
    end
    methods
        %This is a constructor that creates a statistics object to handle
        %our data. We need to feed it with network size and number of common nodes.
        function obj = activation_statistics(act,signal,net_size,n_pure,n_common,thresh,n_real)
            obj.mean_active = mean(act);
            obj.n_pure=n_pure;
            obj.n_common=n_common;
            obj.thresh=thresh;
            % The signal vs time measurements of all the nodes in the
            % network
            obj.signal_time=signal;
            x=n_pure+n_common;

            
            % Here we make up the activation dat
            %obj.act_data(1).ma_matrix=zeros(x,n_real);
            %obj.act_data(1).ma_network_size=zeros(net_size,1);
            %obj.act_data(1).ma_common=zeros(n_common,1);


            obj.ma_matrix=zeros(x,n_real);
            % Mean activity as the function of network size
            obj.ma_network_size=zeros(net_size,1);
            obj.ma_size_r = zeros(net_size,1);
            obj.counter_size=zeros(net_size,1);
            obj.std_size=zeros(net_size,1);
            % Mean activity as the function of the common nodes
            if n_common~=0
                obj.ma_common=zeros(n_common,1);
                obj.counter_common=zeros(n_common,1);
                obj.ma_common_r=zeros(n_common,1);
                obj.ma_common_mat=zeros(n_common,n_real);
                obj.std_common=zeros(n_common,1);
            else
                obj.ma_common=0;
                obj.counter_common=0;
                obj.ma_common_r=0;
                obj.ma_common_mat=0;
                obj.std_common=0;
                
            end
           
        end
        function obj=collectingdata(obj,sig,act,net_size,n_common,thresh,realiz)
            
            obj.mean_active=mean(act);
            obj.signal_time=sig;
            obj.thresh=thresh;

            % ma_matrix is the mean activity matrix vs the realization
            % It captures the mean activity from one realization and stores
            % it in the 2x2 array size where the rows are the net_size and
            % the columss are the realization number 
            obj.ma_matrix(net_size,realiz)=obj.mean_active;

            % ma_network_size is the mean_activity for that realization
            % stored against the network size
            obj.ma_network_size(net_size)=obj.ma_network_size(net_size)+obj.mean_active;
            %obj.ma_std_size(net_size)=st

            % counter_size is probably storing the number of times a
            % certain activity fell in that particular network size. Its
            % going to then find the average mean activity for that network
            % size
            obj.counter_size(net_size)=obj.counter_size(net_size)+1;

            % This is the average mean activity over the ensembles
            obj.ma_size_r(net_size)=obj.ma_network_size(net_size)/obj.counter_size(net_size);
            for i=1:length(obj.ma_matrix(:,1))
                obj.std_size(i)=std(obj.ma_matrix(i,:));
            end
            % Finding the mean activity of the network as the function of
            % common nodes in the network
            if n_common==0
                obj.flag=1;
            end
            if obj.flag==1
                obj.ma_common(n_common+1)=obj.ma_common(n_common+1)+obj.mean_active;
                obj.counter_common(n_common+1)=obj.counter_common(n_common+1)+1;
                obj.ma_common_r(n_common+1)=obj.ma_common(n_common+1)/obj.counter_common(n_common+1);
                obj.ma_common_mat(n_common+1,realiz)=obj.mean_active;
                for i=1:length(obj.std_common)
                    obj.std_common(i,1)=std(obj.ma_common_mat(n_common+1,:));
                end
            else if n_common~=0 && obj.flag==0
                obj.ma_common(n_common)=obj.ma_common(n_common)+obj.mean_active;
                obj.counter_common(n_common)=obj.counter_common(n_common)+1;
                obj.ma_common_r(n_common)=obj.ma_common(n_common)/obj.counter_common(n_common);
                obj.ma_common_mat(n_common,realiz)=obj.mean_active;
                for i=1:length(obj.std_common)
                    obj.std_common(i,1)=std(obj.ma_common_mat(n_common,:));
                end
            
            end
            end
        end
        function plotter(obj,ob,val)
            %obj - Its the crosstalk object
            %ob - Its the non crosstalk object
            switch(val)
               
                % This case plots the mean activity as a function of
                % network size
                case 1
                    
                    x=length(obj.ma_size_r);
                    figure("Name",'Mean activity vs Network Size');
                    boxplot(2:x,obj.ma_size_r(2:x,1));%,obj.std_size(2:x,1));
                    xlabel("Network Size");
                    ylabel("Mean Fraction of Active Nodes");
                
                % This case plots the mean activity as a function of number of common nodes    
                case 2
                    %x=length(obj.ma_common_r);
                    x=0:2;
                    
                    bar(x,obj.ma_common_r);
                    hold on

                        er = errorbar(x,obj.ma_common_r,obj.std_common);
                        er.Color = [0 0 0];                            
                        er.LineStyle = 'none';  
                        
                    hold off
                    
                   
                    %plot(obj.ma_common_r);
                    xlabel("Number of common nodes");
                    ylabel("Mean Fraction of Active Nodes");
                    
                    
                case 3
                    % This is the signal vs time 
                    figure("Name",'Signal Concentration vs time');
                  
                    for i=1:length(obj.signal_time(:,1))
                        if i<=obj.n_pure && obj.n_common==0
                            yline(obj.thresh); hold on;
                            plot(obj.signal_time(i,:));
                        
                        elseif i<=obj.n_pure && obj.n_common >0
                            yline(obj.thresh); hold on;
                            plot(obj.signal_time(i,:),'Color','b');
                        else
                            yline(obj.thresh); hold on;
                            plot(obj.signal_time(i,:),'Color','r');
                        end
                        hold on;
                    end
                    if obj.n_common~=0
                        legend('Blue - Pure Node','Red- Common Node','Black - Signal Threshold');
                    else
                        
                    end
                      
                    xlabel("Time");
                    ylabel("Signal Concentration");

                case 4
                    x=length(obj.ma_size_r);
                    y=length(ob.ma_size_r);
                    figure("Name",'Mean activity vs Network Size');

                    %boxplot(2:x,obj.ma_size_r(2:x,1)); hold on;
                    %boxplot(2:y,ob.ma_size_r(2:y,1));
                    errorbar(2:x,obj.ma_size_r(2:x,1),obj.std_size(2:x,1)); hold on;
                    errorbar(2:y,ob.ma_size_r(2:y,1),ob.std_size(2:y,1));
                    xlabel("Network Size");
                    ylabel("Mean Fraction of Active Nodes");
                    legend('No-Crosstalk','Cross-Talk');
            end
        end
    end
end