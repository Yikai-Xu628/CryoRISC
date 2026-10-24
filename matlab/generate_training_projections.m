function generate_training_projections(volumes, output_root)
    names = {'emd33026','emd70734','emd22689','emd20841','emd3225'};
    seeds = [20260810 20260820 20260910 20260920 20260930];
    for q=1:numel(names)
        name=names{q};
        assert(isfield(volumes,name), ['Missing volume path: ' name]);
        if q<=2
            groups=[300 0 0]; samples=[20 0 0];
        else
            groups=[200 0 0]; samples=[20 0 0];
        end
        generate_grouped_emd33026_dataset('VolumePath',volumes.(name), ...
            'OutputDir',fullfile(output_root,name),'Groups',groups, ...
            'SamplesPerGroup',samples,'RadiusDeg',5,'ProjectionSize',128, ...
            'Seeds',seeds(q)+[0 1 2],'CenterSampling','fibonacci','Overwrite',false);
    end
end
