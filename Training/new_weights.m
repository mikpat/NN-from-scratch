function [w_new, error] = new_weights( w, delta, X, z, n)

    error=0;
    one_vec = ones(size(X,1),1);
    for j = 1:size(w, 2)
        delta_current = delta{j};
        if j==1
            w_new{j} = w{j} - n*[one_vec X]'*delta_current(2:end,:)';
        elseif j == size(w, 2)
            w_new{j} = w{j} - n*[one_vec z{j-1}]'*delta_current';
        else        
            w_new{j} = w{j} - n*[one_vec z{j-1}]'*delta_current(2:end,:)';
        end
    end
    
    for i=1:size(w_new,2)
        if not(isfinite(w_new{i}))
            error = 1;
        end
    end
end

