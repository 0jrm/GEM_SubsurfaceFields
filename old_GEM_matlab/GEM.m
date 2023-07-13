plot_sorted_data = true;
plot_gem_refs = true;
plot_verify = true;
save_figures = false;

load('Subset_data.mat')

%% Sort plot

% sort by steric height

ssh1950 = ARGO_SH1950/100; %cm to m

[sh1950_sorted, idx] = sort(ssh1950);

temp = ARGO_TEMP(:,idx);
sal = ARGO_SAL(:,idx);
pres = -ARGO_PRES(:,1); %all columns are equal, invert sign so plot looks good

if (plot_sorted_data)
    %     plot temperature
    name = 'Temperature';
    srted = " - Sorted Raw Profiles";
    xlbl = "SSH [m]";
    ylbl = "Pressure [dbar]";
    limits = [sh1950_sorted(1) sh1950_sorted(end) -1500 0];

    f1 = figure;
    subplot(2,1,1)
    t = pcolor(sh1950_sorted, pres, temp);
    axis(limits);
    set(t, 'EdgeColor', 'none');
    clim([5 30]);
    colorbar('Ticks',linspace(5,30,5))
    clb1 = colorbar;
    ylabel(clb1, "Temp. [ºC]");
    title(name + srted);
    xlabel(xlbl);
    ylabel(ylbl);

    %     now plot salinity
    subplot(2,1,2)
    name = 'Salinity';
    s = pcolor(sh1950_sorted, pres, sal);
    axis(limits)
    set(s, 'EdgeColor', 'none');
    clim([34 37]);
    ticks = linspace(33.5, 37, 0.5);
    colorbar('Ticks',ticks)
    clb2 = colorbar;
    ylabel(clb2, "Sal. [ºC]");
    title(name + srted);
    xlabel(xlbl);
    ylabel(ylbl);
    if (save_figures)

        saveas(gcf, "Sort_profiles.fig");
        saveas(gcf, "Sort_profiles.jpg");
        close(f1);
    end
end

%% Generating GEM fields

% sample around 80% of data to generate fields
sample_GEM = randsample(length(ARGO_SSH),floor(length(ARGO_SSH)*0.8));
% get the rest for validation
sample_val = setxor(1:length(ARGO_SSH),sample_GEM);

% gathering validation data...
sal_val = sal(:,sample_val);
temp_val = temp(:,sample_val);
pres_val = pres;
ssh_val = ssh1950(sample_val);
%can get other variables later...

% leaving GEM data
sal = sal(:,sample_GEM);
temp = temp(:,sample_GEM);
ssh1950 = ssh1950(sample_GEM);

%set number of nodes
co = 25;
% Scatterplot on reference depths
samp_sh_c = [ min(ssh1950) sort(randsample(ssh1950,co-2)) max(ssh1950)];
samp_sh_b = linspace(min(ssh1950), max(ssh1950), co);
references = [-5 -25:-25:-1950];
% i=10;
ro = length(references);

cfit_t = zeros(ro,co);
bfit_t = zeros(ro,co);
afit_t = zeros(ro,co);

cfit_s = zeros(ro,co);
bfit_s = zeros(ro,co);
afit_s = zeros(ro,co);

hinterv = 0.025;

for i = 1:length(references)
    refnce=references(i);
    index = find(pres==refnce);

    %     average
    tya = zeros(1,co);
    sya = zeros(1,co);
    for j = 1:co
        i_mean = find(and(ssh1950 > samp_sh_b(j)-hinterv, ssh1950 < samp_sh_b(j)+hinterv));
        tya(j) = mean(temp(index,i_mean),'omitnan');
        sya(j) = mean(sal( index,i_mean),'omitnan');

    end


    %     cubic spline, as per GEM method
    tyc = spline(ssh1950,temp(index,:),samp_sh_c);
    syc = spline(ssh1950,sal(index,:),samp_sh_c);
    %     b-spline
    tyb = pchip(ssh1950,temp,samp_sh_b);
    syb = pchip(ssh1950,sal,samp_sh_b);

    %     %     cubic spline, as per GEM method
    %     tyc = spline(samp_sh_c, tya, samp_sh_b);
    %     syc = spline(samp_sh_c, sya, samp_sh_b);
    %     %     b-spline
    %     tyb = pchip(samp_sh_c, tya, samp_sh_b);
    %     syb = pchip(samp_sh_c, sya, samp_sh_b);

    % ^^store these ^^ !!
    afit_t(i, :) = tya;
    bfit_t(i, :) = tyb;
    cfit_t(i, :) = tyc;

    afit_s(i, :) = sya;
    bfit_s(i, :) = syb;
    cfit_s(i, :) = syc;

    % plots

    name = "Empirical fit - ";

    if (plot_gem_refs)
        fp = figure;
        subplot(2,1,1)
        scatter(ssh1950,temp(index,:), 8,'k');
        hold on
        plot(samp_sh_c,tyc, 'r','LineWidth',1.5)
        plot(samp_sh_b,tyb, 'g','LineWidth',1.5)
        plot(samp_sh_b,tya, 'c','LineWidth',1.5)

        dpt = -refnce;
        title(name + refnce + " m");
        ylabel("Temp. [ºC]");

        subplot(2,1,2)
        scatter(ssh1950,sal(index,:),8, 'k');
        hold on
        plot(samp_sh_c,syc, 'r','LineWidth',1.5)
        plot(samp_sh_b,syb, 'g','LineWidth',1.5)
        plot(samp_sh_b,sya, 'c','LineWidth',1.5)

        xlabel("SSH [m]");
        ylabel("Sal. [psu]");
        legend({'Data','Cubic spline', 'B-spline', 'Average'},'Location','best')

        fname = "fir" + dpt;

        if (save_figures)

            saveas(fp, fname + ".fig");
            saveas(fp, fname + ".jpg");
            close(fp);
        end

        clear fp;
    end

end



%% GEM fields -  Plots

%temperature

figure
hAx=arrayfun(@(ix) subplot(1,3,ix),1:3);     % make three subplots

references(1) = 0;
subplot(1,3,1)
[GEMat , gat] = contourf(samp_sh_b,references, afit_t,7,"ShowText",true,"LabelFormat","%0.0f");
grid on
subtitle("Averages")
ylabel("Pressure [dbar]");

subplot(1,3,2)
[GEMbt , gbt] = contourf(samp_sh_b,references, bfit_t,7,"ShowText",true,"LabelFormat","%0.0f");
grid on

title("Temperature GEM field");
subtitle("B-spline")
xlabel("SSH [m]")

subplot(1,3,3)
[GEMct , gct] = contourf(samp_sh_c,references, cfit_t,7,"ShowText",true,"LabelFormat","%0.0f");
grid on
subtitle("Cubic spline")


clb1 = colorbar;
ylabel(clb1, "Temp. [ºC]");
%adjusting presentation

hF=gcf;                                      % get figure handle
set(hAx(2:3),{'YTickLabel'},{[]})    % remove only labels leaving ticks
pos=get(hAx,'position');                     % return the positions
rt=pos{3}(1)+pos{3}(3)-pos{1}(1);       % rightmost axis RH end position
delt=pos{2}(1)-(pos{1}(1)+pos{1}(3));   % delta between 2 and 1; 3 and 2
delt=delt/4;                            % let's halve the present spacing
wnew=pos{1}(3)+delt;                    % so make the new width for all 3
pos{1}(3)=wnew;
pos{2}(3)=wnew;
pos{3}(3)=wnew;
pos{2}(1)=pos{1}(1)+pos{1}(3)+delt/2;   % set LH position of second,
pos{3}(1)=pos{2}(1)+pos{2}(3)+delt/2;   % third; split delta to match
set(hAx(1),'Position',pos{1})           % now set the three new positons
set(hAx(2),'Position',pos{2})
set(hAx(3),'Position',pos{3})


%Salinity

figure
hAx=arrayfun(@(ix) subplot(1,3,ix),1:3);     % make three subplots

subplot(1,3,1)
[GEMas , gas] = contourf(samp_sh_b,references, afit_s,7,"ShowText",true,"LabelFormat","%0.1f");
grid on
subtitle("Averages")
ylabel("Pressure [dbar]");

subplot(1,3,2)
[GEMbs , gbs] = contourf(samp_sh_b,references, bfit_s,7,"ShowText",true,"LabelFormat","%0.1f");
grid on

title("Salinity GEM field");
subtitle("B-spline")
xlabel("SSH [m]")

subplot(1,3,3)
[GEMcs , gcs] = contourf(samp_sh_c,references, cfit_s,7,"ShowText",true,"LabelFormat","%0.1f");
grid on
subtitle("Cubic spline")


clb1 = colorbar;
ylabel(clb1, "Sal. [psu]");
%adjusting presentation

hF=gcf;                                      % get figure handle
set(hAx(2:3),{'YTickLabel'},{[]})    % remove only labels leaving ticks
pos=get(hAx,'position');                     % return the positions
rt=pos{3}(1)+pos{3}(3)-pos{1}(1);       % rightmost axis RH end position
delt=pos{2}(1)-(pos{1}(1)+pos{1}(3));   % delta between 2 and 1; 3 and 2
delt=delt/4;                            % let's halve the present spacing
wnew=pos{1}(3)+delt;                    % so make the new width for all 3
pos{1}(3)=wnew;
pos{2}(3)=wnew;
pos{3}(3)=wnew;
pos{2}(1)=pos{1}(1)+pos{1}(3)+delt/2;   % set LH position of second,
pos{3}(1)=pos{2}(1)+pos{2}(3)+delt/2;   % third; split delta to match
set(hAx(1),'Position',pos{1})           % now set the three new positons
set(hAx(2),'Position',pos{2})
set(hAx(3),'Position',pos{3})


%% Comparison against direct measurements

% %remember validation data...
%
% sal_val = sal(:,sample_val);
% temp_val = temp(:,sample_val);
% pres_val = pres;
% ssh_val = ARGO_SSH(sample_val)/100;

%                           ssh       pres      T or S (from spline or mean)
%[GEMat , gat] = contourf(samp_sh_b,references, afit_t
% ^ repeat for a, b and c. Also t and s ^

% finding range to compare
step = 0.025;
min_range = step + ceil(max([min(ssh_val) min(samp_sh_b)]) * 100) /100;
max_range = -step + floor(min([max(ssh_val) max(samp_sh_b)])  * 100) /100;
%
% ssh_comp_range = min_range:max_range:2*step;

% picking ssh scale used on
samp_ssh = sort(samp_sh_b(and(samp_sh_b>min_range, samp_sh_b<max_range)));
step = min(diff(samp_ssh))/2;

for i=1:length(samp_ssh)
    ref_ssh = samp_ssh(i);
    idx = find(and(ssh_val>ref_ssh-step/2,  ssh_val<ref_ssh+step/2));
    igem = find(and(samp_sh_b>ref_ssh-step/7,  samp_sh_b<ref_ssh+step/7));
    if(~isempty(idx) && ~isempty(igem) && plot_verify)
        % I want to compare model VS real data
        iref = find(samp_sh_b==ref_ssh); % gives x index


        fp = figure;
        set(fp,'Position',[50 50 1500 1500])
        hAx=arrayfun(@(ix) subplot(1,2,ix),1:2);     % make three subplots


        subplot(1,2,1)
        hold on
        plot(afit_t(:, igem), references,'r','LineWidth',1)
        plot(bfit_t(:, igem), references,'g','LineWidth',1)
        plot(cfit_t(:, igem), references,'c','LineWidth',1)
        title("Validation - SH: "+ ref_ssh)
        subtitle("Temperature");
        xlabel("Temp. [°C]")
        ylabel("Pressure [dbar]");

        subplot(1,2,2)
        hold on
        plot(afit_s(:, igem), references,'r','LineWidth',1.5)
        plot(bfit_s(:, igem), references,'g','LineWidth',1.5)
        plot(cfit_s(:, igem), references,'c','LineWidth',1.5)
        set(hAx(2),{'YTickLabel'},{[]})    % remove only labels leaving ticks
        title("Salinity");
        xlabel("Sal. [psu]")

        for j = 1:length(idx)

            subplot(1,2,1)
            hold on
            plot(temp_val(:,idx), pres_val, 'k','LineWidth',0.4)

            legend({'Average', 'B-spline','Cubic spline','Data' },'Location','best')

            subplot(1,2,2)
            hold on

            plot(sal_val(:,idx), pres_val, 'k','LineWidth',0.4)
        end



        if(save_figures)

            fname = "Comp_profile_" + ref_ssh;
            saveas(fp, fname + ".fig");
            saveas(fp, fname + ".jpg");
            close(fp);
        end


    end
end


