function generate_grouped_emd33026_dataset(varargin)

    default_volume = '';
    default_output = '';

    p = inputParser;
    addParameter(p, 'VolumePath', default_volume);
    addParameter(p, 'OutputDir', default_output);
    addParameter(p, 'Groups', [50 10 10]);
    addParameter(p, 'SamplesPerGroup', 20);
    addParameter(p, 'RadiusDeg', 5);
    addParameter(p, 'ProjectionSize', 128);
    addParameter(p, 'Seeds', [20260730 20260731 20260801]);
    addParameter(p, 'CenterSampling', 'random');
    addParameter(p, 'Overwrite', false);
    parse(p, varargin{:});
    cfg = p.Results;
    if isscalar(cfg.SamplesPerGroup)
        samples_per_split = repmat(cfg.SamplesPerGroup, 1, 3);
    elseif numel(cfg.SamplesPerGroup) == 3
        samples_per_split = cfg.SamplesPerGroup;
    else
        error('SamplesPerGroup must be a scalar or [train val test].');
    end

    if isempty(cfg.OutputDir), error('Specify OutputDir explicitly'); end
    if ~exist(cfg.VolumePath, 'file')
        error('Missing volume: %s', cfg.VolumePath);
    end
    if ~exist(cfg.OutputDir, 'dir')
        mkdir(cfg.OutputDir);
    end

    names = {'train', 'val', 'test'};
    files = cell(1, 3);
    for s = 1:3
        files{s} = fullfile(cfg.OutputDir, sprintf( ...
            'emd33026_grouped_%s.mat', names{s}));
        if cfg.Groups(s) == 0, continue; end
        if exist(files{s}, 'file') && ~cfg.Overwrite
            error('Output exists and Overwrite=false: %s', files{s});
        end
    end

    vol = ReadMRC(cfg.VolumePath);
    for s = 1:3
        if cfg.Groups(s) == 0, continue; end
        rng(cfg.Seeds(s), 'twister');
        n_groups = cfg.Groups(s);
        m = samples_per_split(s);
        n = n_groups * m;
        if strcmpi(cfg.CenterSampling, 'fibonacci')
            center_rots = fibonacci_projective_rots(n_groups);
        elseif strcmpi(cfg.CenterSampling, 'random')
            center_rots = rand_rots(n_groups);
        else
            error('CenterSampling must be random or fibonacci.');
        end
        rots = zeros(3, 3, n);
        group_ids = zeros(n, 1);
        center_viewdirs = reshape(permute(center_rots(3, :, :), [2 3 1]), ...
            3, n_groups);
        angular_offsets_deg = zeros(n, 1);

        idx = 0;
        for g = 1:n_groups
            R0 = center_rots(:, :, g);
            v = R0(3, :)';
            for j = 1:m
                idx = idx + 1;
                cos_delta = 1 - rand() * (1 - cosd(cfg.RadiusDeg));
                delta = acos(cos_delta);
                tangent_angle = 2*pi*rand();
                a = R0(1, :)' * cos(tangent_angle) + ...
                    R0(2, :)' * sin(tangent_angle);
                Q = axis_angle_rotation(a, delta);
                gamma = 2*pi*rand();
                Rgamma = [cos(gamma), -sin(gamma), 0; ...
                          sin(gamma),  cos(gamma), 0; ...
                          0,           0,          1];
                rots(:, :, idx) = Rgamma * R0 * Q';
                group_ids(idx) = g;
                angular_offsets_deg(idx) = delta * 180/pi;
            end
        end

        projs = cryo_project(vol, rots, cfg.ProjectionSize, 'single', 100);
        projs = single(permute(projs, [2 1 3]));
        viewdirs = reshape(permute(rots(3, :, :), [2 3 1]), 3, n);
        particle_ids = (1:n)';
        split_name = names{s};
        metadata = struct('volume_path', cfg.VolumePath, ...
            'radius_deg', cfg.RadiusDeg, ...
            'n_groups', n_groups, 'samples_per_group', m, ...
            'projection_size', cfg.ProjectionSize, 'seed', cfg.Seeds(s), ...
            'center_sampling', cfg.CenterSampling, ...
            'viewdir_convention', 'third row of rots passed to cryo_project');

        temp = [files{s} '.partial'];
        save(temp, 'projs', 'rots', 'viewdirs', 'group_ids', ...
            'center_viewdirs', 'angular_offsets_deg', 'particle_ids', ...
            'split_name', 'metadata', '-v7.3');
        movefile(temp, files{s}, 'f');
        fprintf('Saved %s: %d groups x %d = %d images\n', ...
            files{s}, n_groups, m, n);
    end
end

function rots = fibonacci_projective_rots(n)
    golden_angle = pi * (3 - sqrt(5));
    index = (0:n-1)';
    z = (index + 0.5) / n;
    radius = sqrt(max(0, 1 - z.^2));
    phase = 2*pi*rand();
    phi = phase + golden_angle * index;
    directions = [radius .* cos(phi), radius .* sin(phi), z]';
    global_rotation = rand_rots(1);
    directions = global_rotation * directions;

    rots = zeros(3, 3, n);
    for q = 1:n
        v = directions(:, q) / norm(directions(:, q));
        helper = [0; 0; 1];
        if abs(dot(helper, v)) > 0.9
            helper = [0; 1; 0];
        end
        row1 = cross(helper, v); row1 = row1 / norm(row1);
        row2 = cross(v, row1); row2 = row2 / norm(row2);
        rots(:, :, q) = [row1'; row2'; v'];
    end
end


function R = axis_angle_rotation(axis, angle)
    axis = axis / norm(axis);
    x = axis(1); y = axis(2); z = axis(3);
    K = [0, -z, y; z, 0, -x; -y, x, 0];
    R = eye(3) + sin(angle)*K + (1-cos(angle))*(K*K);
end
