function solvefrac(param)
    param = param/10000;
    load('forcing.mat',"F");
    load('mesh.mat',"X");
    Nsample = size(F,2);
    U = zeros(size(F));
    for i = 1:Nsample
        % Matrix of double derivative
        f = chebfun(F(:,i), [-1,1], 'trig');
        n = length(f);
        % Ensure n is odd
        if mod(n,2) == 0
            n = n+1;
        end
        
        % Define differentiation matrix of -d^(2s)/d^2s on [-1,1]
        Dx2 = (-1*trigspec.diffmat(n,2)).^param;
        % Add zero mean condition
        Dx2(floor(n/2)+1,floor(n/2)+1) = 1;
        
        % Solve equation
        fx = trigcoeffs(f, n);
        uc = Dx2 \ fx;
        
        % Convert Fourier coeffs to values
        u = trigtech.coeffs2vals(uc);
        
        % Make chebfun
        u = chebfun(u, f.domain, 'trig');
        
        for j = 1:size(X,1)
            U(j,i) = u(X(j));
        end
    end
    save('solution.mat',"U");
end

