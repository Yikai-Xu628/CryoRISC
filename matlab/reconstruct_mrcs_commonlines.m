function reconstruct_mrcs_commonlines(input_mrcs, output_mrc, checkpoint_mat, log_file, apix)
    if nargin < 5
        error('Specify the physical pixel size explicitly.');
    end
    diary(log_file);
    cleanup = onCleanup(@() diary('off'));
    projs = single(ReadMRC(input_mrcs));
    n = size(projs, 1); K = size(projs, 3); n_theta = 360;
    fprintf('Reconstruction input: %d x %d x %d\n', n, n, K);
    for k = 1:K
        im = projs(:, :, k);
        projs(:, :, k) = im - mean(im(:));
    end
    npf = cryo_pft(projs, n, n_theta, 'single');
    if exist('cryo_clmatrix_cpu', 'file')
        [clstack, corrstack, shift_equations] = cryo_clmatrix_cpu(npf, -1, 1, 5, 1);
    else
        [clstack, corrstack, shift_equations] = cryo_clmatrix(npf, -1, 1, 5, 1, 0);
    end
    corrstack = corrstack + corrstack';
    A = [shift_equations(:, 1:end-1); ...
         sparse(1:3, 1:3, ones(1, 3), 3, 2*K); eye(2*K)];
    b = [shift_equations(:, end); 0; 0; 0; zeros(2*K, 1)];
    est_shifts = full(reshape(A \ b, 2, K));
    S = cryo_syncmatrix_vote(clstack, n_theta);
    est_rotssyn = cryo_syncrotations(S);
    save(checkpoint_mat, 'clstack', 'corrstack', 'shift_equations', ...
         'est_shifts', 'S', 'est_rotssyn', 'input_mrcs', '-v7.3');
    vol = recon3d_firm(projs, est_rotssyn, -est_shifts', 1e-6, 20, zeros(n,n,n));
    WriteMRC(single(real(vol)), apix, output_mrc);
    fprintf('Saved reconstruction: %s\n', output_mrc);
end
