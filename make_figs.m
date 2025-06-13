close all
clear all
clc

L=3;
lag=1;


%% Fig. 4: Compare to Boaretto et al raw


boa_init_r1 = readmatrix(['eeg_processed/ensemble_linear_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
boa_init_r2 = readmatrix(['eeg_processed/ensemble_linear_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
boa_best_r1 = readmatrix(['eeg_processed/ensemble_best_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
boa_best_r2 = readmatrix(['eeg_processed/ensemble_best_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
h_r1 = readmatrix(['eeg_processed/ensemble_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
h_r2 = readmatrix(['eeg_processed/ensemble_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
v_r1 = readmatrix(['eeg_processed/ensemble_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
v_r2 = readmatrix(['eeg_processed/ensemble_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 


boa_init_f1 = readmatrix(['eeg_processed/ensemble_linear_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
boa_init_f2 = readmatrix(['eeg_processed/ensemble_linear_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 
boa_best_f1 = readmatrix(['eeg_processed/ensemble_best_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
boa_best_f2 = readmatrix(['eeg_processed/ensemble_best_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 
h_f1 = readmatrix(['eeg_processed/ensemble_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
h_f2 = readmatrix(['eeg_processed/ensemble_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 
v_f1 = readmatrix(['eeg_processed/ensemble_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
v_f2 = readmatrix(['eeg_processed/ensemble_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 


c=[0.6350,0.0780,0.1840; 0    0.4470    0.7410];

figure,subplot(2,1,1), hold on, set(gca,'XLim',[0.5,4.5])
errorbar(1,mean(boa_init_r2),std(boa_init_r2),'LineWidth',2,'Color',c(2,:))
errorbar(2,mean(boa_init_r1),std(boa_init_r1),'LineWidth',2,'Color',c(1,:))
plot(1,mean(boa_init_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(2,mean(boa_init_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

fill([2.5  2.5 4.5 4.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([6.5  6.5 8.5 8.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')

errorbar(3,mean(boa_best_r2),std(boa_best_r2),'LineWidth',2,'Color',c(2,:))
errorbar(4,mean(boa_best_r1),std(boa_best_r1),'LineWidth',2,'Color',c(1,:))
plot(3,mean(boa_best_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(4,mean(boa_best_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(5,mean(h_r2),std(h_r2),'Color',c(2,:),'LineWidth',2)
errorbar(6,mean(h_r1),std(h_r1),'Color',c(1,:),'LineWidth',2)
plot(5,mean(h_r2),'sb','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(6,mean(h_r1),'sr','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(7,mean(v_r2),std(v_r2),'Color',c(2,:),'LineWidth',2)
errorbar(8,mean(v_r1),std(v_r1),'Color',c(1,:),'LineWidth',2)
plot(7,mean(v_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(8,mean(v_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

ylabel('$\langle H^s_t\rangle_t$','Interpreter','latex')
%ylabel('$\langle SPE^i(t)\rangle_t$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)



 row2 = {'EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','Boaretto et al. initial', ' ',' ',' '  , 'Boaretto et al. best', ' ',' ',' '  ,'Horizontal Symbols', ' ',' ',' '  ,'Vertical Symbols', ' ' };
% %row3 = 10.5:13.5; 
 labelArray = [row2; row1]; 
 tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
 ax = gca(); 
 ax.XTick =1:.5:8; 
 ax.XLim = [0,5];
  ax.YLim = [.87,.99];
  ax.YTick=.9:.03:.99;
 %ax.XTickLabel = tickLabels; 
 
 set(gca,'XTickLabelRotatio',0,'XLim',[.5,8.5])
 set(gca,'Layer','top','GridAlpha',.5)
box on, grid on

% Compare to Boaretto et al filt

subplot(2,1,2), hold on, set(gca,'XLim',[0.5,4.5])
errorbar(1,mean(boa_init_f2),std(boa_init_f2),'Color',c(2,:),'LineWidth',2)
errorbar(2,mean(boa_init_f1),std(boa_init_f1),'Color',c(1,:),'LineWidth',2)
plot(1,mean(boa_init_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(2,mean(boa_init_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

fill([2.5  2.5 4.5 4.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([6.5  6.5 8.5 8.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')


errorbar(3,mean(boa_best_f2),std(boa_best_f2),'Color',c(2,:),'LineWidth',2)
errorbar(4,mean(boa_best_f1),std(boa_best_f1),'Color',c(1,:),'LineWidth',2)
plot(3,mean(boa_best_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(4,mean(boa_best_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(5,mean(h_f2),std(h_f2),'Color',c(2,:),'LineWidth',2)
errorbar(6,mean(h_f1),std(h_f1),'Color',c(1,:),'LineWidth',2)
plot(5,mean(h_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(6,mean(h_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(7,mean(v_f2),std(v_f2),'Color',c(2,:),'LineWidth',2)
errorbar(8,mean(v_f1),std(v_f1),'Color',c(1,:),'LineWidth',2)
plot(7,mean(v_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(8,mean(v_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

ylabel('$\langle H^s_t\rangle_t$','Interpreter','latex')
%ylabel('$\langle SPE^i(t)\rangle_t$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
 set(gca,'Layer','top','GridAlpha',.5)


 row2 = {'EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','     linear', ' ',' ',' '  , '  alternative', ' ',' ',' '  ,'horizontal', ' ',' ',' '  ,' vertical', ' ' };
 row3 = {' ','arrangement', ' ',' ',' '  , 'arrangement', ' ',' ',' '  ,' symbols', ' ',' ',' ' ,' symbols', ' ' };
 
 labelArray = [row2; row1; row3]; 
 tickLabels = strtrim(sprintf('%s\\newline%s\\newline%s\n', labelArray{:}));
 ax = gca(); 
 ax.XTick = 1:.5:8; 
 ax.XLim = [0,5];
 ax.YLim=[.75,0.97];
 ax.XTickLabel = tickLabels; 
 %set(gca,'YTickLabel',{'0.70','0,80','0,90','1,0'})
 
 set(gca,'XTickLabelRotatio',0,'XLim',[.5,8.5])
set(subplot(2,1,1),'Position',[0.1300 0.60 0.7750 0.3700])
set(subplot(2,1,2),'Position',[0.1300 0.22 0.7750 0.37])
b_text=text(-.7,0.97,'b)','FontName','Helvetica', 'FontSize',18);
a_text=text(-.7,1.2,'a)','FontName','Helvetica', 'FontSize',18);
grid on, box on

saveas(gcf,'fig4','epsc')


%% Fig. 5: Hotizontal vs. vertical

avSPEt_r1 = readmatrix(['eeg_processed/spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
avSPEt_r2 = readmatrix(['eeg_processed/spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
avSPEt_f1 = readmatrix(['eeg_processed/spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
avSPEt_f2 = readmatrix(['eeg_processed/spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

v_avSPEt_r1 = readmatrix(['eeg_processed/spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
v_avSPEt_r2 = readmatrix(['eeg_processed/spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
v_avSPEt_f1 = readmatrix(['eeg_processed/spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
v_avSPEt_f2 = readmatrix(['eeg_processed/spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

stdSPEt_r1 = readmatrix(['eeg_processed/std_spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
stdSPEt_r2 = readmatrix(['eeg_processed/std_spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
stdSPEt_f1 = readmatrix(['eeg_processed/std_spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
stdSPEt_f2 = readmatrix(['eeg_processed/std_spe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

v_stdSPEt_r1 = readmatrix(['eeg_processed/std_spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
v_stdSPEt_r2 = readmatrix(['eeg_processed/std_spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
v_stdSPEt_f1 = readmatrix(['eeg_processed/std_spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
v_stdSPEt_f2 = readmatrix(['eeg_processed/std_spe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

new_PSPE_h_r1 = readmatrix(['eeg_processed/pspe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
new_PSPE_h_r2 = readmatrix(['eeg_processed/pspe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
new_PSPE_h_f1 = readmatrix(['eeg_processed/pspe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
new_PSPE_h_f2 = readmatrix(['eeg_processed/pspe_hor_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

new_PSPE_v_r1 = readmatrix(['eeg_processed/pspe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
new_PSPE_v_r2 = readmatrix(['eeg_processed/pspe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
new_PSPE_v_f1 = readmatrix(['eeg_processed/pspe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
new_PSPE_v_f2 = readmatrix(['eeg_processed/pspe_ver_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 


c=[0.6350,0.0780,0.1840; 0    0.4470    0.7410];
%figure,% sgtitle('raw')
figure,
tt=tiledlayout(3,1);
tt.TileSpacing = 'tight';

nexttile
hold on, set(gca,'XLim',[0.25,4.25])
errorbar(1,mean(avSPEt_r1),std(avSPEt_r1),'Color',c(1,:),'LineWidth',2)
plot(1,mean(avSPEt_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(.5,mean(avSPEt_r2),std(avSPEt_r2),'Color',c(2,:),'LineWidth',2)
plot(.5,mean(avSPEt_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))


fill([1.25  1.25 2.25 2.25],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([3.25  3.25 4.25 4.25],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')

errorbar(2,mean(avSPEt_f1),std(avSPEt_f1),'Color',c(1,:),'LineWidth',2)
plot(2,mean(avSPEt_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(1.5,mean(avSPEt_f2),std(avSPEt_f2),'Color',c(2,:),'LineWidth',2)
plot(1.5,mean(avSPEt_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(3,mean(v_avSPEt_r1),std(v_avSPEt_r1),'Color',c(1,:),'LineWidth',2)
plot(3,mean(v_avSPEt_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(2.5,mean(v_avSPEt_r2),std(v_avSPEt_r2),'Color',c(2,:),'LineWidth',2)
plot(2.5,mean(v_avSPEt_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(4,mean(v_avSPEt_f1),std(v_avSPEt_f1),'Color',c(1,:),'LineWidth',2)
plot(4,mean(v_avSPEt_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(3.5,mean(v_avSPEt_f2),std(v_avSPEt_f2),'Color',c(2,:),'LineWidth',2)
plot(3.5,mean(v_avSPEt_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))



ylabel('$\langle H^s_t\rangle_t$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
set(gca,'XTick',[.5:.25:4],'XTickLabel',{'','','',''})
box on, grid on
set(gca,'YLim',[.68,1],'YTick',.7:.1:1)


 set(gca,'Layer','top','GridAlpha',.5)

nexttile, hold on, set(gca,'XLim',[0.25,4.25])

errorbar(1,mean(stdSPEt_r1),std(stdSPEt_r1),'Color',c(1,:),'LineWidth',2)
plot(1,mean(stdSPEt_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(.5,mean(stdSPEt_r2),std(stdSPEt_r2),'Color',c(2,:),'LineWidth',2)
plot(.5,mean(stdSPEt_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))


fill([1.25  1.25 2.25 2.25],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([3.25  3.25 4.25 4.25],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')

errorbar(2,mean(stdSPEt_f1),std(stdSPEt_f1),'Color',c(1,:),'LineWidth',2)
plot(2,mean(stdSPEt_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(1.5,mean(stdSPEt_f2),std(stdSPEt_f2),'Color',c(2,:),'LineWidth',2)
plot(1.5,mean(stdSPEt_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(3,mean(v_stdSPEt_r1),std(v_stdSPEt_r1),'Color',c(1,:),'LineWidth',2)
plot(3,mean(v_stdSPEt_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(2.5,mean(v_stdSPEt_r2),std(v_stdSPEt_r2),'Color',c(2,:),'LineWidth',2)
plot(2.5,mean(v_stdSPEt_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(4,mean(v_stdSPEt_f1),std(v_stdSPEt_f1),'Color',c(1,:),'LineWidth',2)
plot(4,mean(v_stdSPEt_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(3.5,mean(v_stdSPEt_f2),std(v_stdSPEt_f2),'Color',c(2,:),'LineWidth',2)
plot(3.5,mean(v_stdSPEt_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))


ylabel('$\sigma \left( H^s_t\right)$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
set(gca,'XTick',[.5:.25:4],'XTickLabel',{'','','',''})

box on, grid on
set(gca,'Layer','top','GridAlpha',.5)
set(gca, 'YLim',[0,.201], 'YTick',0:.1:.2)
nexttile
hold on, set(gca,'XLim',[0.25,4.25])
errorbar(.5,mean(new_PSPE_h_r2),std(new_PSPE_h_r2),'Color',c(2,:),'LineWidth',2)
plot(.5,mean(new_PSPE_h_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(1,mean(new_PSPE_h_r1),std(new_PSPE_h_r1),'Color',c(1,:),'LineWidth',2)
plot(1,mean(new_PSPE_h_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))



fill([1.25  1.25 2.25 2.25],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([3.25  3.25 4.25 4.25],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')

errorbar(1.5,mean(new_PSPE_h_f2),std(new_PSPE_h_f2),'Color',c(2,:),'LineWidth',2)
plot(1.5,mean(new_PSPE_h_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(2,mean(new_PSPE_h_f1),std(new_PSPE_h_f1),'Color',c(1,:),'LineWidth',2)
plot(2,mean(new_PSPE_h_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(2.5,mean(new_PSPE_v_r2),std(new_PSPE_v_r2),'Color',c(2,:),'LineWidth',2)
plot(2.5,mean(new_PSPE_v_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(3,mean(new_PSPE_v_r1),std(new_PSPE_v_r1),'Color',c(1,:),'LineWidth',2)
plot(3,mean(new_PSPE_v_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(3.5,mean(new_PSPE_v_f2),std(new_PSPE_v_f2),'Color',c(2,:),'LineWidth',2)
plot(3.5,mean(new_PSPE_v_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(4,mean(new_PSPE_v_f1),std(new_PSPE_v_f1),'Color',c(1,:),'LineWidth',2)
plot(4,mean(new_PSPE_v_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

ylabel('$H^s_{pi}$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
%set(gca,'XTick',[.5:.25:4],'XTickLabel',{'','','',''})
box on, grid on

row2 = {'EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','raw', ' ',' ',' ', 'filt.', ' ',' ',' ','raw', ' ',' ',' ', 'filt.', ' ',};
row3 = {' ', ' ',' ','horizontal symbols',' ', ' ', ' ',' ',' ',' ', ' ','vertical symbols',' ', ' ', ' ',};

labelArray = [ row2; row1; row3]; 
tickLabels = strtrim(sprintf('%s\\newline%s\\newline%s\n', labelArray{:}));



 set(gca,'Layer','top','GridAlpha',.5)

 ax = gca(); 
 ax.XTick =.5:.25:4; 
 ax.YTick =.85:.05:1; 
 ax.XLim = [.25,4.25];
  ax.YLim = [.85,1];
 ax.XTickLabel = tickLabels;
ax.XTickLabelRotation=0;

set(gcf,'Position',[277 259 560.0000 488])
a_text=text(-.4,0.46,'a)','FontName','Helvetica', 'FontSize',18);
b_text=text(-.4,0.275,'b)','FontName','Helvetica', 'FontSize',18);
c_text=text(-.4,0.1,'c)','FontName','Helvetica', 'FontSize',18);

saveas(gcf,'fig5','epsc')

%% Fig 6: Plot temporal features

av_PE_r1_L3_lag1 = readmatrix(['eeg_processed/pe_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
av_PE_r2_L3_lag1 = readmatrix(['eeg_processed/pe_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
av_PE_f1_L3_lag1 = readmatrix(['eeg_processed/pe_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
av_PE_f2_L3_lag1 = readmatrix(['eeg_processed/pe_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

std_PE_r1_L3_lag1 = readmatrix(['eeg_processed/pe_std_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
std_PE_r2_L3_lag1 = readmatrix(['eeg_processed/pe_std_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
std_PE_f1_L3_lag1 = readmatrix(['eeg_processed/pe_std_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
std_PE_f2_L3_lag1 = readmatrix(['eeg_processed/pe_std_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

PPE_r1_L3_lag1 = readmatrix(['eeg_processed/ppe_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']);
PPE_r2_L3_lag1 = readmatrix(['eeg_processed/ppe_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
PPE_f1_L3_lag1 = readmatrix(['eeg_processed/ppe_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']);
PPE_f2_L3_lag1 = readmatrix(['eeg_processed/ppe_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 



c=[0.6350,0.0780,0.1840; 0    0.4470    0.7410];
%figure,% sgtitle('raw')
figure
t=tiledlayout(3,1);
t.TileSpacing = 'tight';

%subplot(3,1,1),
nexttile
hold on, set(gca,'XLim',[0.25,2.25])
errorbar(1,mean(av_PE_r1_L3_lag1),std(av_PE_r1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(.5,mean(av_PE_r2_L3_lag1),std(av_PE_r2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(1,mean(av_PE_r1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(.5,mean(av_PE_r2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

fill([1.25  1.25 2.25 2.25],[0.6 1 1 0.6],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')



errorbar(2,mean(av_PE_f1_L3_lag1),std(av_PE_f1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(1.5,mean(av_PE_f2_L3_lag1),std(av_PE_f2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(2,mean(av_PE_f1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(1.5,mean(av_PE_f2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

set(gca,'Layer','top','GridAlpha',.5)
ylabel('$\langle H^s_i\rangle_i$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
set(gca,'XTick',[.5:.25:2],'XTickLabel',{'','','',''})
box on, grid on

%subplot(3,1,2), 
nexttile
hold on, set(gca,'XLim',[0.25,2.25])
errorbar(1,mean(std_PE_r1_L3_lag1),std(std_PE_r1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(.5,mean(std_PE_r2_L3_lag1),std(std_PE_r2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(1,mean(std_PE_r1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(.5,mean(std_PE_r2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
fill([1.25  1.25 2.25 2.25],[0 .04 .04 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')

errorbar(2,mean(std_PE_f1_L3_lag1),std(std_PE_f1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(1.5,mean(std_PE_f2_L3_lag1),std(std_PE_f2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(2,mean(std_PE_f1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(1.5,mean(std_PE_f2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))


ylabel('$\sigma \left( H^s_i\right)$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
set(gca,'XTick',[.5:.25:2],'XTickLabel',{'','','',''})
box on, grid on


set(gca,'Layer','top','GridAlpha',.5)

%subplot(3,1,3),
nexttile
hold on, set(gca,'XLim',[0.5,2.5])
errorbar(1,mean(PPE_r1_L3_lag1),std(PPE_r1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(.5,mean(PPE_r2_L3_lag1),std(PPE_r2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(1,mean(PPE_r1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(.5,mean(PPE_r2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
fill([1.25  1.25 2.25 2.25],[0.6 1 1 0.6],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')

errorbar(2,mean(PPE_f1_L3_lag1),std(PPE_f1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(1.5,mean(PPE_f2_L3_lag1),std(PPE_f2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(2,mean(PPE_f1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(1.5,mean(PPE_f2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))



ylabel('$H^s_{pt}$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
%set(gca,'XTick',[1,2],'XTickLabel',{'EO','EC'})
box on, grid on

 row2 = {'EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','raw', ' ',' ',' ', 'filtered', ' '};
% %row3 = 10.5:13.5; 
 labelArray = [row2; row1]; 
 tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
 ax = gca(); 
 ax.XTick =.5:.25:2; 
 ax.XLim = [.25,2.25];
%  ax.YLim = [.87,.99];
 ax.XTickLabel = tickLabels;
ax.XTickLabelRotation=0;
set(gca,'Layer','top','GridAlpha',.5)
set(gcf,"Position",[277 259 560 488])
a_text=text(-.09,1.99,'a)','FontName','Helvetica', 'FontSize',18);
b_text=text(-.09,1.49,'b)','FontName','Helvetica', 'FontSize',18);
c_text=text(-.09,1.01,'c)','FontName','Helvetica', 'FontSize',18);

saveas(gcf,'fig6','epsc')

%% Fig 7: All methods comparison

boat_best_sub_r1=readmatrix(['eeg_processed/spe_boa_L_' num2str(L) '_lag_' num2str(lag) '_run_1_raw.csv']); 
boat_best_sub_r2=readmatrix(['eeg_processed/spe_boa_L_' num2str(L) '_lag_' num2str(lag) '_run_2_raw.csv']); 
boat_best_sub_f1=readmatrix(['eeg_processed/spe_boa_L_' num2str(L) '_lag_' num2str(lag) '_run_1_filt.csv']); 
boat_best_sub_f2=readmatrix(['eeg_processed/spe_boa_L_' num2str(L) '_lag_' num2str(lag) '_run_2_filt.csv']); 

c=[0.6350,0.0780,0.1840; 0    0.4470    0.7410];

figure,subplot(2,1,1), hold on, set(gca,'XLim',[-1.5,4.5])

fill([-1.5  -1.5 0.5 0.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([2.5  2.5 4.5 4.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([6.5  6.5 8.5 8.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([10.5  10.5 12.5 12.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')


errorbar(0,mean(av_PE_r1_L3_lag1),std(av_PE_r1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(-1,mean(av_PE_r2_L3_lag1),std(av_PE_r2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(0,mean(av_PE_r1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(-1,mean(av_PE_r2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(1,mean(avSPEt_r2),std(avSPEt_r2),'LineWidth',2,'Color',c(2,:))
errorbar(2,mean(avSPEt_r1),std(avSPEt_r1),'LineWidth',2,'Color',c(1,:))
plot(1,mean(avSPEt_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(2,mean(avSPEt_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(3,mean(v_avSPEt_r2),std(v_avSPEt_r2),'Color',c(2,:),'LineWidth',2)
errorbar(4,mean(v_avSPEt_r1),std(v_avSPEt_r1),'Color',c(1,:),'LineWidth',2)
plot(3,mean(v_avSPEt_r2),'sb','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(4,mean(v_avSPEt_r1),'sr','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(5,mean(boat_best_sub_r2),std(boat_best_sub_r2),'Color',c(2,:),'LineWidth',2)
errorbar(6,mean(boat_best_sub_r1),std(boat_best_sub_r1),'Color',c(1,:),'LineWidth',2)
plot(5,mean(boat_best_sub_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(6,mean(boat_best_sub_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(7,mean(new_PSPE_h_r2),std(new_PSPE_h_r2),'Color',c(2,:),'LineWidth',2)
errorbar(8,mean(new_PSPE_h_r1),std(new_PSPE_h_r1),'Color',c(1,:),'LineWidth',2)
plot(7,mean(new_PSPE_h_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(8,mean(new_PSPE_h_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(9,mean(new_PSPE_v_r2),std(new_PSPE_v_r2),'Color',c(2,:),'LineWidth',2)
errorbar(10,mean(new_PSPE_v_r1),std(new_PSPE_v_r1),'Color',c(1,:),'LineWidth',2)
plot(9,mean(new_PSPE_v_r2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(10,mean(new_PSPE_v_r1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(11,mean(PPE_r2_L3_lag1),std(PPE_r2_L3_lag1),'Color',c(2,:),'LineWidth',2)
errorbar(12,mean(PPE_r1_L3_lag1),std(PPE_r1_L3_lag1),'Color',c(1,:),'LineWidth',2)
plot(11,mean(PPE_r2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(12,mean(PPE_r1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

ylabel('Permutation Entropy')
%ylabel('$\langle SPE^i(t)\rangle_t$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)



 row2 = {'EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','Boaretto et al. initial', ' ',' ',' '  , 'Boaretto et al. best', ' ',' ',' '  ,'Horizontal Symbols', ' ',' ',' '  ,'Vertical Symbols', ' ' };
% %row3 = 10.5:13.5; 
 labelArray = [row2; row1]; 
 tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
 ax = gca(); 
 ax.XTick =-1:.5:12; 
 ax.XLim = [0,5];
  ax.YLim = [.82,1];
  ax.YTick=.85:.05:1;
 %ax.XTickLabel = tickLabels; 
 
 set(gca,'XTickLabelRotatio',0,'XLim',[-1.5,12.5],'XTickLabel',{' ',' ', ' ',' ',' ' })
 set(gca,'Layer','top','GridAlpha',.5)
box on, grid on

% Compare to Boaretto et al filt

subplot(2,1,2), hold on, set(gca,'XLim',[0.5,4.5])

fill([-1.5  -1.5 0.5 0.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([2.5  2.5 4.5 4.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([6.5  6.5 8.5 8.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([10.5  10.5 12.5 12.5],[0 1 1 0],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')


errorbar(0,mean(av_PE_f1_L3_lag1),std(av_PE_f1_L3_lag1),'Color',c(1,:),'LineWidth',2)
errorbar(-1,mean(av_PE_f2_L3_lag1),std(av_PE_f2_L3_lag1),'Color',c(2,:),'LineWidth',2)
plot(0,mean(av_PE_f1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
plot(-1,mean(av_PE_f2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))

errorbar(1,mean(avSPEt_f2),std(avSPEt_f2),'LineWidth',2,'Color',c(2,:))
errorbar(2,mean(avSPEt_f1),std(avSPEt_f1),'LineWidth',2,'Color',c(1,:))
plot(1,mean(avSPEt_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(2,mean(avSPEt_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(3,mean(v_avSPEt_f2),std(v_avSPEt_f2),'Color',c(2,:),'LineWidth',2)
errorbar(4,mean(v_avSPEt_f1),std(v_avSPEt_f1),'Color',c(1,:),'LineWidth',2)
plot(3,mean(v_avSPEt_f2),'sb','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(4,mean(v_avSPEt_f1),'sr','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(5,mean(boat_best_sub_f2),std(boat_best_sub_f2),'Color',c(2,:),'LineWidth',2)
errorbar(6,mean(boat_best_sub_f1),std(boat_best_sub_f1),'Color',c(1,:),'LineWidth',2)
plot(5,mean(boat_best_sub_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(6,mean(boat_best_sub_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(7,mean(new_PSPE_h_f2),std(new_PSPE_h_f2),'Color',c(2,:),'LineWidth',2)
errorbar(8,mean(new_PSPE_h_f1),std(new_PSPE_h_f1),'Color',c(1,:),'LineWidth',2)
plot(7,mean(new_PSPE_h_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(8,mean(new_PSPE_h_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(9,mean(new_PSPE_v_f2),std(new_PSPE_v_f2),'Color',c(2,:),'LineWidth',2)
errorbar(10,mean(new_PSPE_v_f1),std(new_PSPE_v_f1),'Color',c(1,:),'LineWidth',2)
plot(9,mean(new_PSPE_v_f2),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(10,mean(new_PSPE_v_f1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(11,mean(PPE_f2_L3_lag1),std(PPE_f2_L3_lag1),'Color',c(2,:),'LineWidth',2)
errorbar(12,mean(PPE_f1_L3_lag1),std(PPE_f1_L3_lag1),'Color',c(1,:),'LineWidth',2)
plot(11,mean(PPE_f2_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(2,:),'Color',c(2,:))
plot(12,mean(PPE_f1_L3_lag1),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))


y_lab=ylabel('Permutation Entropy');
y_lab.Position(1)=-2.8781;
y_lab.Position(2)=0.81;
y_lab.Position(3)=1;
%ylabel('$\langle SPE^i(t)\rangle_t$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)
 set(gca,'Layer','top','GridAlpha',.5)

row2 = {'EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','A', ' ',' ',' ' ,'B(hor.)', ' ',' ',' '  ,'B(vert.)', ' ',' ',' '  ,'B(alt.)', ' ',' ',' '  ,'C(hor.)', ' ',' ',' '  ,'C(vert.)', ' ',' ',' '  ,'D',' ' };
 row3 = {' ',' ', ' ',' ',' '  , '(hor.)', ' ',' ',' '  ,'(vert.)', ' ',' ',' ' ,'(alt.)' ' ',' ',' ' ,'(hor.)' ' ',' ',' ' ,'(vert.)' ' ',' ',' ' ,'(alt.)',' ' };
 
 labelArray = [row2; row1;];


 tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
 ax = gca(); 
 ax.XTick = -1:.5:12; 
 ax.XLim = [0,5];
 ax.YLim=[.63,1];
 ax.XTickLabel = tickLabels; 
 %set(gca,'YTickLabel',{'0.70','0,80','0,90','1,0'})
 
 set(gca,'XTickLabelRotatio',0,'XLim',[-1.5,12.5])
 set(gcf,'Position',[455 236 560 518])
 set(subplot(2,1,1),'Position',[0.1300 0.55 0.8 0.4])
 set(subplot(2,1,2),'Position',[0.1300 0.11 0.8 0.4])
b_text=text(-3.7,1.01,'b)','FontName','Helvetica', 'FontSize',18);
a_text=text(-3.7,1.42,'a)','FontName','Helvetica', 'FontSize',18);
grid on, box on

saveas(gcf,'fig7','epsc')

%% Fig.8: Relative Entropy Difference

c=[0.6350,0.0780,0.1840; 0    0.4470    0.7410];

figure,subplot(2,1,1), hold on, set(gca,'XLim',[-1.5,4.5])

fill([-1.5  -1.5 0.5 0.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([2.5  2.5 4.5 4.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([6.5  6.5 8.5 8.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([10.5  10.5 12.5 12.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
plot([-5,25],[0,0],'--','Color','k','LineWidth',2)


errorbar(-0.5,mean(2.*(av_PE_r1_L3_lag1-av_PE_r2_L3_lag1)./(av_PE_r1_L3_lag1+av_PE_r2_L3_lag1)),std(2.*(av_PE_r1_L3_lag1-av_PE_r2_L3_lag1)./(av_PE_r1_L3_lag1+av_PE_r2_L3_lag1)),'Color',c(1,:),'LineWidth',2)
plot(-.5,mean(2.*(av_PE_r1_L3_lag1-av_PE_r2_L3_lag1)./(av_PE_r1_L3_lag1+av_PE_r2_L3_lag1)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(1.5,mean(2.*(avSPEt_r1-avSPEt_r2)./(avSPEt_r1+avSPEt_r2)),std(2.*(avSPEt_r1-avSPEt_r2)./(avSPEt_r1+avSPEt_r2)),'LineWidth',2,'Color',c(1,:))
plot(1.5,mean(2.*(avSPEt_r1-avSPEt_r2)./(avSPEt_r1+avSPEt_r2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(3.5,mean(2.*(v_avSPEt_r1-v_avSPEt_r2)./(v_avSPEt_r1+v_avSPEt_r2)),std(2.*(v_avSPEt_r1-v_avSPEt_r2)./(v_avSPEt_r1+v_avSPEt_r2)),'Color',c(1,:),'LineWidth',2)
plot(3.5,mean(2.*(v_avSPEt_r1-v_avSPEt_r2)./(v_avSPEt_r1+v_avSPEt_r2)),'sr','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(5.5,mean(2.*(boat_best_sub_r1-boat_best_sub_r2)./(boat_best_sub_r1+boat_best_sub_r2)),std(2.*(boat_best_sub_r1-boat_best_sub_r2)./(boat_best_sub_r1+boat_best_sub_r2)),'Color',c(1,:),'LineWidth',2)
plot(5.5,mean(2.*(boat_best_sub_r1-boat_best_sub_r2)./(boat_best_sub_r1+boat_best_sub_r2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(7.5,mean(2.*(new_PSPE_h_r1-new_PSPE_h_r2)./(new_PSPE_h_r1+new_PSPE_h_r2)),std(2.*(new_PSPE_h_r1-new_PSPE_h_r2)./(new_PSPE_h_r1+new_PSPE_h_r2)),'Color',c(1,:),'LineWidth',2)
plot(7.5,mean(2.*(new_PSPE_h_r1-new_PSPE_h_r2)./(new_PSPE_h_r1+new_PSPE_h_r2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(9.5,mean(2.*(new_PSPE_v_r1-new_PSPE_v_r2)./(new_PSPE_v_r1+new_PSPE_v_r2)),std(2.*(new_PSPE_v_r1-new_PSPE_v_r2)./(new_PSPE_v_r1+new_PSPE_v_r2)),'Color',c(1,:),'LineWidth',2)
plot(9.5,mean(2.*(new_PSPE_v_r1-new_PSPE_v_r2)./(new_PSPE_v_r1+new_PSPE_v_r2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(11.5,mean(2.*(PPE_r1_L3_lag1-PPE_r2_L3_lag1)./(PPE_r1_L3_lag1+PPE_r2_L3_lag1)),std(2.*(PPE_r1_L3_lag1-PPE_r2_L3_lag1)./(PPE_r1_L3_lag1+PPE_r2_L3_lag1)),'Color',c(1,:),'LineWidth',2)
plot(11.5,mean(2.*(PPE_r1_L3_lag1-PPE_r2_L3_lag1)./(PPE_r1_L3_lag1+PPE_r2_L3_lag1)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

%ylabel('$\langle H^s_t\rangle_t$','Interpreter','latex')
%ylabel('$\langle SPE^i(t)\rangle_t$','Interpreter','latex')
set(gca,'FontName','Helvetica', 'FontSize',18)

ylabel('Relative entropy difference')

 row2 = {'EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','Boaretto et al. initial', ' ',' ',' '  , 'Boaretto et al. best', ' ',' ',' '  ,'Horizontal Symbols', ' ',' ',' '  ,'Vertical Symbols', ' ' };
% %row3 = 10.5:13.5; 
 labelArray = [row2; row1]; 
 tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
 ax = gca(); 
 ax.XTick =-1:.5:12; 
 ax.XLim = [0,5];
  ax.YLim = [-0.06,0.06];
  ax.YTick=-.05:.025:.05;
 ax.XTickLabel = {' ', ' ', ' '}; 
 
 set(gca,'XTickLabelRotatio',0,'XLim',[-1.5,12.5])
 set(gca,'Layer','top','GridAlpha',.5)
box on, grid on

% Compare to Boaretto et al filt

subplot(2,1,2), hold on, set(gca,'XLim',[0.5,4.5])

fill([-1.5  -1.5 0.5 0.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([2.5  2.5 4.5 4.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([6.5  6.5 8.5 8.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')
fill([10.5  10.5 12.5 12.5],[-1 1 1 -1],'k', 'FaceAlpha',.1,'LineStyle','None','EdgeColor','w')

plot([-5,25],[0,0],'--','Color','k','LineWidth',2)

errorbar(-0.5,mean(2.*(av_PE_f1_L3_lag1-av_PE_f2_L3_lag1)./(av_PE_f1_L3_lag1+av_PE_f2_L3_lag1)),std(2.*(av_PE_f1_L3_lag1-av_PE_f2_L3_lag1)./(av_PE_f1_L3_lag1+av_PE_f2_L3_lag1)),'Color',c(1,:),'LineWidth',2)
plot(-0.5,mean(2.*(av_PE_f1_L3_lag1-av_PE_f2_L3_lag1)./(av_PE_f1_L3_lag1+av_PE_f2_L3_lag1)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(1.5,mean(2.*(avSPEt_f1-avSPEt_f2)./(avSPEt_f1+avSPEt_f2)),std(2.*(avSPEt_f1-avSPEt_f2)./(avSPEt_f1+avSPEt_f2)),'LineWidth',2,'Color',c(1,:))
plot(1.5,mean(2.*(avSPEt_f1-avSPEt_f2)./(avSPEt_f1+avSPEt_f2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(3.5,mean(2.*(v_avSPEt_f1-v_avSPEt_f2)./(v_avSPEt_f1+v_avSPEt_f2)),std(2.*(v_avSPEt_f1-v_avSPEt_f2)./(v_avSPEt_f1+v_avSPEt_f2)),'Color',c(1,:),'LineWidth',2)
plot(3.5,mean(2.*(v_avSPEt_f1-v_avSPEt_f2)./(v_avSPEt_f1+v_avSPEt_f2)),'sr','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(5.5,mean(2.*(boat_best_sub_f1-boat_best_sub_f2)./(boat_best_sub_f1+boat_best_sub_f2)),std(2.*(boat_best_sub_f1-boat_best_sub_f2)./(boat_best_sub_f1+boat_best_sub_f2)),'Color',c(1,:),'LineWidth',2)
plot(5.5,mean(2.*(boat_best_sub_f1-boat_best_sub_f2)./(boat_best_sub_f1+boat_best_sub_f2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(7.5,mean(2.*(new_PSPE_h_f1-new_PSPE_h_f2)./(new_PSPE_h_f1+new_PSPE_h_f2)),std(2.*(new_PSPE_h_f1-new_PSPE_h_f2)./(new_PSPE_h_f1+new_PSPE_h_f2)),'Color',c(1,:),'LineWidth',2)
plot(7.5,mean(2.*(new_PSPE_h_f1-new_PSPE_h_f2)./(new_PSPE_h_f1+new_PSPE_h_f2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(9.5,mean(2.*(new_PSPE_v_f1-new_PSPE_v_f2)./(new_PSPE_v_f1+new_PSPE_v_f2)),std(2.*(new_PSPE_v_f1-new_PSPE_v_f2)./(new_PSPE_v_f1+new_PSPE_v_f2)),'Color',c(1,:),'LineWidth',2)
plot(9.5,mean(2.*(new_PSPE_v_f1-new_PSPE_v_f2)./(new_PSPE_v_f1+new_PSPE_v_f2)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))

errorbar(11.5,mean(2.*(PPE_f1_L3_lag1-PPE_f2_L3_lag1)./(PPE_f1_L3_lag1+PPE_f2_L3_lag1)),std(2.*(PPE_f1_L3_lag1-PPE_f2_L3_lag1)./(PPE_f1_L3_lag1+PPE_f2_L3_lag1)),'Color',c(1,:),'LineWidth',2)
plot(11.5,mean(2.*(PPE_f1_L3_lag1-PPE_f2_L3_lag1)./(PPE_f1_L3_lag1+PPE_f2_L3_lag1)),'s','MarkerSize',10,'MarkerFaceColor',c(1,:),'Color',c(1,:))
y_lab=ylabel('Relative entropy difference');
y_lab.Position(1)=-3.4094;
y_lab.Position(2)=0.1;
y_lab.Position(3)=1;

set(gca,'FontName','Helvetica', 'FontSize',18)
 set(gca,'Layer','top','GridAlpha',.5)


row2 = {'EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO',' ','EC',' ','EO'};
 row1 = {' ','A', ' ',' ',' ' ,'B(hor.)', ' ',' ',' '  ,'B(vert.)', ' ',' ',' '  ,'B(alt.)', ' ',' ',' '  ,'C(hor.)', ' ',' ',' '  ,'C(vert.)', ' ',' ',' '  ,'D',' ' };
 row3 = {' ',' ', ' ',' ',' '  , '(hor.)', ' ',' ',' '  ,'(vert.)', ' ',' ',' ' ,'(alt.)' ' ',' ',' ' ,'(hor.)' ' ',' ',' ' ,'(vert.)' ' ',' ',' ' ,'(alt.)',' ' };
 
 labelArray = [row2; row1;];


 tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
 ax = gca(); 
 ax.XTick = -1:.5:12; 
 ax.XLim = [0,5];
 ax.YLim=[-0.01,.2];
 ax.XTickLabel = row1;%tickLabels; 
 %set(gca,'YTickLabel',{'0.70','0,80','0,90','1,0'})
 
 set(gca,'XTickLabelRotatio',0,'XLim',[-1.5,12.5])
 set(gcf,'Position',[455 181 560 573])
 set(subplot(2,1,1),'Position',[0.1800 0.55 0.8 0.4])
 set(subplot(2,1,2),'Position',[0.1800 0.11 0.8 0.4])
a_text=text(-4.5,0.45,'a)','FontName','Helvetica', 'FontSize',18);
b_text=text(-4.5,0.2,'b)','FontName','Helvetica', 'FontSize',18);
grid on, box on

saveas(gcf,'fig8','epsc')

