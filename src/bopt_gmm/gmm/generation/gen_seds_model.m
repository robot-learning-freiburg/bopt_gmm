function [Priors, Mu, Sigma] = get_seds_model(pwd, x0, xT, Data, o_index, n_priors, objective, dt, tol_cutting, max_iter)

    % display(pwd)
    % display(x0)
    % display(xT)
    % display(Data)
    % display(o_index)
    % display(n_priors)
    % display(objective)
    
    % Pre-processing
    % dt = 0.1; %The time step of the demonstrations
    % tol_cutting = .05; % A threshold on velocity that will be used for trimming demos
    
    % Training parameters
    K = n_priors; %Number of Gaussian funcitons
    
    % A set of options that will be passed to the solver. Please type 
    % 'doc preprocess_demos' in the MATLAB command window to get detailed
    % information about other possible options.
    options.tol_mat_bias = 10^-6; % A very small positive scalar to avoid
                                  % instabilities in Gaussian kernel [default: 10^-15]
                                  
    options.display = 1;          % An option to control whether the algorithm
                                  % displays the output of each iterations [default: true]
                                  
    options.tol_stopping=10^-10;  % A small positive scalar defining the stoppping
                                  % tolerance for the optimization solver [default: 10^-10]
    
    options.max_iter = max_iter;  % Maximum number of iteration for the solver [default: i_max=1000]
    
    options.objective = objective;    % 'likelihood': use likelihood as criterion to
                                  % optimize parameters of GMM
                                  % 'mse': use mean square error as criterion to
                                  % optimize parameters of GMM
                                  % 'direction': minimize the angle between the
                                  % estimations and demonstrations (the velocity part)
                                  % to optimize parameters of GMM                              
                                  % [default: 'mse']
    
    %% Putting GMR and SEDS library in the MATLAB Path
    if isempty(regexp(path,['SEDS_lib' pathsep], 'once'))
        addpath([pwd, '/SEDS_lib']);    % add SEDS dir to path
    end
    if isempty(regexp(path,['GMR_lib' pathsep], 'once'))
        addpath([pwd, '/GMR_lib']);    % add GMR dir to path
    end
    
    %% SEDS learning algorithm
    % [x0 , xT, Data, index] = preprocess_demos(demos,dt,tol_cutting); %preprocessing datas
    [Priors_0, Mu_0, Sigma_0] = initialize_SEDS(Data,K); %finding an initial guess for GMM's parameter
    [Priors Mu Sigma]=SEDS_Solver(Priors_0,Mu_0,Sigma_0,Data,options); %running SEDS optimization solver
    end
    
    