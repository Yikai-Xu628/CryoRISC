function results = compare_reference_fsc(reference, volumes, methods, apix)
    assert(numel(volumes)==numel(methods), 'One label per volume is required');
    vol = ReadMRC(reference);
    n = size(vol,1);
    x = (1:n/2)/(n*apix);
    thresholds = [0.5, 0.143];
    resolution = nan(numel(volumes),2);
    figure('Position',[100,100,1000,550]); hold on;
    colors = {'r-','b-','g-'};
    fprintf('\n%-20s %15s %15s\n','Method','FSC=0.5 (A)','FSC=0.143 (A)');
    for i=1:numel(volumes)
        candidate = ReadMRC(volumes{i});
        assert(isequal(size(vol),size(candidate)), 'Volume grids differ');
        ff = FSCorr(vol,candidate); ff = ff(:); xx = x(:);
        assert(numel(ff)==numel(xx), 'FSC length differs from frequency bins');
        plot(xx,ff,colors{mod(i-1,numel(colors))+1},'LineWidth',1.5);
        for j=1:2
            t=thresholds(j);
            k=find(ff(1:end-1)>=t & ff(2:end)<t,1,'first');
            if ~isempty(k)
                f=xx(k)+(t-ff(k))*(xx(k+1)-xx(k))/(ff(k+1)-ff(k));
                resolution(i,j)=1/f;
            end
        end
        fprintf('%-20s %15.3f %15.3f\n',methods{i},resolution(i,1),resolution(i,2));
    end
    legend(methods,'FontSize',20,'FontName','Times New Roman','Location','northeast');
    plot(x,ones(size(x))*.143,'k:',x,ones(size(x))*.5,'k:','HandleVisibility','off');
    set(gca,'FontSize',16,'FontName','Times New Roman');
    xlim([0 max(x)]); ylim([-.2 1.01]);
    xlabel('Spatial frequency (1/\r{A})','Interpreter','latex','FontSize',16);
    ylabel('FSC','FontSize',16,'FontName','Times New Roman');
    results=table(methods(:),resolution(:,1),resolution(:,2), ...
        'VariableNames',{'Method','Resolution_at_05_A','Resolution_at_0143_A'});
end
