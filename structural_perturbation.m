classdef structural_perturbation
    % This class creates object that can handle two neural network. It
    % understands the relation between the two neural networks in the form
    % of the matrix node_matrix. Using that information it is able to
    % perturb them 

    % Structural perturbation class has the tools to calculate hamming
    % distance and store it and use it for finding the different
    % relationships 

    % Hamming Distance
    

    properties
        
        % Inputting the two evolved Neural network to be perturbed

        pert_nn;            % This is the network that is going to be perturbed                            
        unpert_nn;          % This is the unperturbed ntwork
        
        final_pert_nn;
        final_unpert_nn;
        node_matrix;        % This matrix shows the relation between the two neural networks 
        
        norm_hd_pertnn;     %This is normalized hamming distance of the perturbed network
        norm_hd_unpert;     % This is the normalized hamming distance of the unperturbed network
        

        pert_size;          % This entails how much structural perturbation we are giving to the system
                   
        pert_name;          % What kind of perturbation is this
        pert_target;        % What is the target quantity in this case 
                            % Target == 1 means signal concentration perturbation
                            % Target == 2 means cell number concnetration perturbation
                            % Target == 3 means network size perturbations

        time;
        delta_t;
        fold_change;

    end

    methods
        function obj = structural_perturbation(nn1,nn2,node_matrix,pert_name,pert_target,pert_radius,time,delta_t,f)

            obj.pert_nn=nn1;
            obj.unpert_nn=nn2;
            obj.final_pert_nn=[];
            obj.final_unpert_nn=[];
            obj.node_matrix=node_matrix;
            obj.pert_name=pert_name;
            obj.pert_target=pert_target; % This tells me what to perturb
                                         % is it signal or cell number or just the network size
            
            % Storing the perturbation as the percentage from -100 to +100%
            x=randi(2*pert_radius+1,1);
           
            if x<=pert_radius
                obj.pert_size=0-x/10;
            else 
                obj.pert_size=[(x-pert_radius-1)/10];
            end

            obj.time=time;
            obj.delta_t=delta_t;
            obj.fold_change=f;

        end


        % Here the number of common nodes is fixed !
        % Here we are perturbing any random pure node and see how it
        % affects the NN
        function [n_h1,n_h2,pert_obj] = perturb_commonnode(pert_obj)
            
           pert_nn=pert_obj_1.pert_nn;
           unpert_nn=pert_obj_1.unpert_nn;


           x = pert_nn.n_common; % Total pure nodes in the perturbed NN
           %x2 = obj.nn2.n_pure;  %[x1+1, x1+x2] indices of the 2nd layer
           com=randi(x);
           node_to_perturb_1=pert_nn.n_pure+com; % Selecting a random common node in pert layer
           node_to_perturb_2=unpert_nn.n_pure+com;
           


           switch(pert_obj.pert_target) % perturb the signal concentration vector
               
               case 1
                        pert_nn.signal(node_to_perturb_1)=pert_nn.signal(node_to_perturb_1)*(1+pert_obj.pert_size);
                        unpert_nn.signal(node_to_perturb_2)=unpert_nn.signal(node_to_perturb_2)*(1+pert_obj.pert_size);

               case 2
                   if pert_obj.pert_size >= -1
                        pert_nn.cell_pop(node_to_perturb_1)=pert_nn.cell_pop(node_to_perturb_1)*(1+pert_obj.pert_size);
                        unpert_nn.cell_pop(node_to_perturb_2)=unpert_nn.cell_pop(node_to_perturb_2)*(1+pert_obj.pert_size);
                        %pert_obj.pert_size=0;
                   end
                  

               %case 3
                   %pert_nn.n(node_to_perturb)=pert_nn.cell_pop(node_to_perturb)*(1+pert_size);
           end


           % The time evolution of the network after the perturbation is
           % executed
           [final_pert_nn,final_unpert_nn]=time_series(pert_nn,unpert_nn,pert_obj.node_matrix);

           HD_pert_lay = sum(abs(final_pert_nn.act-pert_nn.act));
           HD_unpert_lay = sum(abs(final_unpert_nn.act-unpert_nn.act));
           if perturb_which_net ==1
               pert_obj.nn1=pert_nn;
               pert_obj.nn2=unpert_nn;
           else
               pert_obj.nn1=unpert_nn;
               pert_obj.nn2=pert_nn;
           end
           
           pert_obj.norm_hd_pertnn=HD_pert_lay/pert_nn.n;
           pert_obj.norm_hd_unpert=HD_unpert_lay/pert_nn.n; 
           n_h1=pert_obj.norm_hd_pertnn;
           n_h2=pert_obj.norm_hd_unpert;
        end

       

        function [n_h1,n_h2,pert_obj] = perturb_purenode(pert_obj)
            
           % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
           % n_h1 : is the normalized HD for the perturbed layer
           % n_h2 : is the normalized HD for the unperturbed layer
           % net_signal : contains the signal concentration vs time for
           % both the network. It also has the previous time-series signal
           % concentraion matrix. It will add the new time-series signal
           % concetration after the perturbation.
           % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


           % Copying the Neural Networks
           pert_nn=pert_obj.pert_nn;
           unpert_nn=pert_obj.unpert_nn;

           % Total pure nodes in the perturbed NN
           x = pert_nn.n_pure; 
           % Selecting a random pure node to be perturbed
           node_to_perturb=randi(x);
           HD_pert=0;
           
           % - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
           % Perturbation target tells me which array to perturb - signal
           % concentration array or the cell number concentration array
           % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           switch(pert_obj.pert_target) % perturb the signal concentration vector
               
               case 1
                   pert_nn.signal(node_to_perturb)=pert_nn.signal(node_to_perturb)*(1+pert_obj.pert_size);
                   [x,y]=size(pert_nn.sig_time);
                   pert_nn.sig_time(:,y)=pert_nn.signal;
               case 2
                   if pert_obj.pert_size >= -1
                        pert_nn.cell_pop(node_to_perturb)=pert_nn.cell_pop(node_to_perturb)*(1+pert_obj.pert_size);

                        % In this special case, the node is basically
                        % deleted, but however the network can still behave
                        % keeping this node activated if the net sum of
                        % weighted signals from the other nodes exceed the
                        % threshold concentration !
                        % Hence deletion of a population at any instant
                        % means that the activity state of that node in the
                        % very next instant = 0
                        if pert_obj.pert_size==-1
                            pert_nn.act(node_to_perturb)=0;
                            HD_pert = sum(abs(pert_obj.pert_nn.act-pert_nn.act));
                        end
                       
                   end
                  

               %case 3
                   %pert_nn.n(node_to_perturb)=pert_nn.cell_pop(node_to_perturb)*(1+pert_size);
           end
           pert_obj.pert_nn=pert_nn;
           pert_obj.unpert_nn=unpert_nn;

           
           pert_obj=time_series(pert_obj);
           

           HD_pert = sum(abs(pert_obj.final_pert_nn.act-pert_obj.pert_nn.act));
           HD_unpert = sum(abs(pert_obj.final_unpert_nn.act-pert_obj.unpert_nn.act));
          
          
           pert_obj.norm_hd_pertnn=HD_pert/pert_nn.n;
           pert_obj.norm_hd_unpert=HD_unpert/unpert_nn.n;
           
           n_h1=pert_obj.norm_hd_pertnn;
           n_h2=pert_obj.norm_hd_unpert;
           
           
        end
 

        function [n_h1,n_h2,pert_obj] = perturb_targetnode(pert_obj,node_id)
            
           node_to_perturb=node_id;
           pert_nn=pert_obj.pert_nn;
           unpert_nn=pert_obj.unpert_nn;
           x1 = pert_nn.n_common; % Total pure nodes in the perturbed NN
           x2 = pert_nn.n_pure;  %[x1+1, x1+x2] indices of the 2nd layer
           y1=unpert_nn.n_pure;
           y2=unpert_nn.n_common;
           HD_pert=0;
           if node_id >x2
               diff=node_id-x2;
           end
           
           % Here we are going to perturb the target node only
           
           


           switch(pert_obj.pert_target) % perturb the signal concentration vector
               
               case 1
                   pert_nn.signal(node_to_perturb)=pert_nn.signal(node_to_perturb)*(1+pert_obj.pert_size);

               case 2
                   if pert_obj.pert_size >= -1
                        pert_nn.cell_pop(node_to_perturb)=pert_nn.cell_pop(node_to_perturb)*(1+pert_obj.pert_size);

                        % In this special case, the node is basically
                        % deleted, but however the network can still behave
                        % keeping this node activated if the net sum of
                        % weighted signals from the other nodes exceed the
                        % threshold concentration !
                        % Hence deletion of a population at any instant
                        % means that the activity state of that node in the
                        % very next instant = 0
                        if pert_obj.pert_size==-1
                            pert_nn.act(node_to_perturb)=0;
                            % Storing the change in hamming distance due
                            % to this perturbation here
                            HD_pert = sum(abs(pert_obj.pert_nn.act-pert_nn.act));
                        end
                        if node_id > x2
                            unpert_nn.cell_pop(y1+diff)=unpert_nn.cell_pop(y1+diff)*(1+pert_obj.pert_size);
                        end
                       
                   end
                  

               %case 3
                   %pert_nn.n(node_to_perturb)=pert_nn.cell_pop(node_to_perturb)*(1+pert_size);
           end
           pert_obj.pert_nn=pert_nn;
           pert_obj.unpert_nn=unpert_nn;

           

           pert_obj=time_series(pert_obj);
           

           HD_pert =HD_pert+ sum(abs(pert_obj.final_pert_nn.act-pert_obj.pert_nn.act));
           HD_unpert = sum(abs(pert_obj.final_unpert_nn.act-pert_obj.unpert_nn.act));
          
          
           pert_obj.norm_hd_pertnn=HD_pert/pert_nn.n;
           pert_obj.norm_hd_unpert=HD_unpert/unpert_nn.n;
           
    
           n_h1=pert_obj.norm_hd_pertnn;
           n_h2=pert_obj.norm_hd_unpert;
        end
    
    
        function [h_pert,h_unpert,net_signal,pert_obj]=perturb_setrandnodes(pert_obj,net_id,r)
            % r= no of random nodes to perturb

            pert_obj.which_net=net_id;

            if net_id ==1
               pert_nn=pert_obj.nn1;
               unpert_nn=pert_obj.nn2;
           else
               pert_nn=pert_obj.nn2;
               unpert_nn=pert_obj.nn1;
           end
           x1 = pert_nn.n_common; % Total pure nodes in the perturbed NN
           x2 = pert_nn.n_pure;  %[x1+1, x1+x2] indices of the 2nd layer
           y1=unpert_nn.n_pure;
           y2=unpert_nn.n_common;
           
           
           node_id=randperm(pert_nn.n,r); % The unique random nodes it is going to perturb
           for i=1:r
                   if node_id(i) >x2
                        diff=node_id(i)-x2;
                   end
                   switch(pert_obj.pert_target) % perturb the signal concentration vector
                   
                   case 1
                            pert_nn.signal(node_id(i))=pert_nn.signal(node_id(i))*(1+pert_obj.pert_size);
                            % When the node we selected  is also a common node
                           
    
                   case 2
                       if pert_obj.pert_size >= -1
                            pert_nn.cell_pop(node_id)=pert_nn.cell_pop(node_id)*(1+pert_obj.pert_size);
                            if node_id > x2
                                unpert_nn.cell_pop(y1+diff)=unpert_nn.cell_pop(y1+diff)*(1+pert_obj.pert_size);
                            end
                            %pert_obj.pert_size=0;
                       end
                   end
           end
            % The time evolution of the network after the perturbation is
           % executed
           [final_pert_nn,final_unpert_nn,net_signal]=time_series(pert_nn,unpert_nn,pert_obj.node_matrix);

           HD_pert_lay = sum(abs(final_pert_nn.act-pert_nn.act));
           HD_unpert_lay = sum(abs(final_unpert_nn.act-unpert_nn.act));

           if net_id ==1
               pert_obj.nn1=pert_nn;
               pert_obj.nn2=unpert_nn;
        
           else
               pert_obj.nn1=unpert_nn;
               pert_obj.nn2=pert_nn;
               net_1=net_signal(1).signal;
               net_2=net_signal(2).signal;
               net_signal(1).signal=net_2; %This is the unperturbed layer
               net_signal(2).signal=net_1; %This is the perturbed layer
           end
           h_pert=HD_pert_lay;
           h_unpert=HD_unpert_lay;
         end
    end

end

function pert_obj = time_series(pert_obj)
    % This time series is similar to the time-evolution function we see in
    % the main code. Except the input in this function is a perturbation
    % object and not a neuralnetwork object


    % The perturbation object is an object of this class. It has two
    % neuralnetwork objects as its properites. So we are going to help to
    % feed this function with those values.

    NN(1)=pert_obj.pert_nn;
    NN(2)=pert_obj.unpert_nn;

    node_matrix=pert_obj.node_matrix;
 
    time=pert_obj.time;
    delta_t=pert_obj.delta_t;
    f=pert_obj.fold_change;
    
    %node_matrix=pert_obj.node_matrix;
    for t=1:time
        for i=1:length(node_matrix)
           
            factor_1=NN(i).sig_p*delta_t*(ones(NN(i).n,1)+(f-1)*NN(i).act);
            factor_2=1-NN(i).sig_d*delta_t;
            NN(i).signal=factor_2*NN(i).signal+factor_1.*NN(i).cell_pop;
            NN(i).sig_time=cat(2,NN(i).sig_time,NN(i).signal);
            
            
      
       
            
            total_input_signal=NN(i).weights.'*NN(i).signal;
            response=total_input_signal-NN(i).h;
            NN(i).act=heaviside(response);

            % Here we are making sure of the fact that if the
            % cell-population is zero then the activity of that node is
            % zero !
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
    end  
    
    pert_obj.final_pert_nn=NN(1);
    pert_obj.final_unpert_nn=NN(2);
  
    
end




