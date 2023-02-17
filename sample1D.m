function sample1D(sigma, Nsample, seed)
    rng(seed)
    sigma = sigma/10000;
    
    load('mesh.mat');
    N = size(X,1);
    
    dom = [min(X), max(X)];
    domain_length = dom(end) - dom(1);
    
    F = zeros(N, Nsample);
    
    K = chebfun2(@(x,y) exp(-(x-y).^2 / (2 * domain_length ^2 * sigma^2)), [dom, dom]);
    L = chol(K,'lower');
    
    for i = 1:Nsample
       f = generate_random_function(L);
       F(:,i) = f(X);
    end
    save('dat1D.mat',"F");
end

function f = generate_random_function(L)
    u = randn(rank(L),1);
    f = L * u;
end