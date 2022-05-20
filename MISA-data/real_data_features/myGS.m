function out = myGS(in)
% Computes Gram-Schimidt decomp.

out = zeros(size(in));
out(:,1) = in(:,1);
out(:,1) = (1/sqrt(out(:,1)'*out(:,1)))*out(:,1); % Normalize

for kk = 2:size(in,2)
    
    tmp = 0;
    
    for jj = 1:(kk-1)
        
        tmp = tmp + proj(in(:,kk),out(:,jj));
        
    end
    
    out(:,kk) = in(:,kk) - tmp;
    out(:,kk) = (1/sqrt(out(:,kk)'*out(:,kk)))*out(:,kk); % Normalize
    
end

end

function out = proj(v,u) % project v onto line spanned by u
    out = ((v'*u)/(u'*u))*u;
end