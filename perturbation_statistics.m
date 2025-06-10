classdef perturbation_statistics
    % This class helps me in reading the data collected from the
    % perturbations of the two NN during a realization.
    % This will caluculate all the graphs required and help me in getting a
    % better and conscise understanding of my system

    properties
        % Here we take in the normalized hamming distances according to the
        % perturbation sizes

        pert_radius;
        pert_size;
        max_entropy;
        %stability;

        % We are going to utilize a structure to create our normalized
        % hamming distances and also store the standard deviations

        pert_data=struct('pert_size',[0],'norm_hd',[0],'en_mean_hd',[0],'freq_pert_size',[0],'std',[0],'distribution',[0],'entropy',[0],'stability',[0]);

        % Measuring the entropy of the 
        
        
    end

    methods
        function obj = perturbation_statistics(net_size,pert_radius)
    
        
          
            % This is keeping a counter of hamming distance vs the
            % perturbation size given to the object
            obj.pert_radius=pert_radius;
            obj.max_entropy=-log(1/net_size);
            %obj.stability=0;
            z=1;
            for x=-pert_radius:1:pert_radius
                    
                obj.pert_size(z)=x/10;
                obj.pert_data(z).pert_size=obj.pert_size(z);
                obj.pert_data(z).norm_hd=0;
                obj.pert_data(z).en_mean_hd=0;
                obj.pert_data(z).freq_pert_size=0;
                obj.pert_data(z).std=0;
                obj.pert_data(z).distribution=tabulate(obj.pert_data(z).norm_hd);
                obj.pert_data(z).entropy=0;
                obj.pert_data(z).stability=0;
                    
                z=z+1;
            end

   
           
        end

        function obj=collecting_data(obj,norm_hd,pert_size)
           pert_id=find(obj.pert_size==pert_size); %It returns the array index of the perturbation size
           
           l=length(obj.pert_data(pert_id).norm_hd);

          
        
           obj.pert_data(pert_id).norm_hd(l+1)=norm_hd;
             
           
           obj.pert_data(pert_id).freq_pert_size=obj.pert_data(pert_id).freq_pert_size+1;
           obj.pert_data(pert_id).en_mean_hd=mean(obj.pert_data(pert_id).norm_hd);
           obj.pert_data(pert_id).std=std(obj.pert_data(pert_id).norm_hd);
            
           % Fiding the entropy of my Hamming distance distribution for
           % each kind of perturbation
           tab=tabulate(obj.pert_data(pert_id).norm_hd);
           
            
           obj.pert_data(pert_id).distribution=tab;
           tab=tab(:,3)/100;

           obj.pert_data(pert_id).entropy=-nansum(times(tab,log(tab)));

           obj.pert_data(pert_id).stability=1-(obj.pert_data(pert_id).entropy)/obj.max_entropy;

         
        end

        function plotter(obj,name,col)
        
            for i=1:2*obj.pert_radius+1
                mean_norm_hd(i)=obj.pert_data(i).en_mean_hd;
                std(i)=obj.pert_data(i).std;
            end
            % Comment this when doing signal perturbation otherwise the
            % other line for cell perturation
            %errorbar(100*obj.pert_size,mean_norm_hd,std,Color=col);
            errorbar(100*obj.pert_size(2:2*obj.pert_radius+1),mean_norm_hd(2:2*obj.pert_radius+1),std(2:2*obj.pert_radius+1),Color=col);
            title(name);
            xlabel('Perturbation in %');
            ylabel('Mean Normalized Hamming Distance');
        end
        function plot_entropy(obj,name,col)
            for i=1:2*obj.pert_radius+1
                entropy(i)=obj.pert_data(i).entropy;
                %std(i)=obj.pert_data(i).std;
            end
            plot(100*obj.pert_size,entropy,Color=col)
           % errorbar(100*obj.pert_size,mean_norm_hd,std,Color=col);
            title(name);
            xlabel('Perturbation in %');
            ylabel('Entropy');
        end
        
        function plot_hd_dist(obj)

            obj.pert_data
            plot(100*obj.pert_size,entropy,Color=col)
           % errorbar(100*obj.pert_size,mean_norm_hd,std,Color=col);
            title(name);
            xlabel('Hamming distance');
            ylabel('Probability');
        end

            
            
            

    end
end