function sample2D(sigma, Nsample, vargin)
    load('mesh2D.mat');
    N = size(X,1);
    
    F = zeros(N, Nsample);
    
    sigma = sigma/10000;
    
    for i = 1:Nsample
        f = randnfundisk(sigma);
        F(:,i) = f(X(:,1),X(:,2));
    end
    save('dat2D.mat',"F");
end


