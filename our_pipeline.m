% complete pipeline for calcium imaging data pre-processing
clear;
time_start = tic;
% addpath(genpath('../NoRMCorre'));               % add the NoRMCorre motion correction package to MATLAB path
gcp;                                            % start a parallel engine
foldername = 'D:\\cygwin64\\home\\USER\\eyes_converge_data\\191118_9\\';
         % folder where all the files are located.
filetype = 'h5'; % type of files to be processed
        % Types currently supported .tif/.tiff, .h5/.hdf5, .raw, .avi, and .mat files
files = subdir(fullfile(foldername,['*.',filetype]));   % list of filenames (will search all subdirectories)
FOV = size(read_file(files(1).name,1,1));
numFiles = length(files);


%% downsample h5 files and save into a single memory mapped matlab file

registered_files = files;
    
fr = 10;                                         % frame rate
tsub = 1;                                        % degree of downsampling (for 30Hz imaging rate you can try also larger, e.g. 8-10)
ds_filename = [foldername,'/ds_data.mat'];       % Delete this temp file before running this script since this temp file will increase every time we run this script.
data_type = class(read_file(registered_files(1).name,1,1));
data = matfile(ds_filename,'Writable',true);
data.Y  = zeros([FOV,0],data_type);
data.Yr = zeros([prod(FOV),0],data_type);
data.sizY = [FOV,0];
F_dark = Inf;                                    % dark fluorescence (min of all data)
batch_size = 10;                               % read chunks of that size
batch_size = round(batch_size/tsub)*tsub;        % make sure batch_size is divisble by tsub
Ts = zeros(numFiles,1);                          % store length of each file
cnt = 0;                                         % number of frames processed so far
% tt1 = tic;
for i = 1:numFiles
    name = registered_files(i).name;
    info = h5info(name);
    dims = info.Datasets.Dataspace.Size;
    ndimsY = length(dims);                       % number of dimensions (data array might be already reshaped)
    Ts(i) = dims(end);
    % Ysub = zeros(FOV(1),FOV(2),FOV(3),floor(Ts(i)/tsub),data_type);
    data.Y(FOV(1),FOV(2),FOV(3),sum(floor(Ts/tsub))) = zeros(1,data_type);
    data.Yr(prod(FOV),sum(floor(Ts/tsub))) = zeros(1,data_type);
    cnt_sub = 0;
    for t = 1:batch_size:Ts(i)
        Y = read_file(name,t,min(batch_size,Ts(i)-t+1));    
        F_dark = min(nanmin(Y(:)),F_dark);
        ln = size(Y,ndimsY);
        Y = reshape(Y,[FOV,ln]);
        Y = cast(downsample_data(Y,'time',tsub),data_type);
        ln = size(Y,4);
        % Ysub(:,:,:,cnt_sub+1:cnt_sub+ln) = Y;
        data.Y(:,:,:,cnt+cnt_sub+1:cnt+cnt_sub+ln) = Y;
        data.Yr(:,cnt+cnt_sub+1:cnt+cnt_sub+ln) = reshape(Y,[],ln);
        cnt_sub = cnt_sub + ln;
    end
    % data.Y(:,:,:,cnt+1:cnt+cnt_sub) = Ysub;
    % data.Yr(:,cnt+1:cnt+cnt_sub) = reshape(Ysub,[],cnt_sub);
    % toc(tt1);
    cnt = cnt + cnt_sub;
    data.sizY(1,4) = cnt;
end
data.F_dark = F_dark;
%% now run CNMF on patches on the downsampled file, set parameters first

sizY = data.sizY;                       % size of data matrix
patch_size = [70,70,70];                   % size of each patch along each dimension (optional, default: [32,32])
overlap = [10,10,10];                        % amount of overlap in each dimension (optional, default: [4,4])

patches = construct_patches(sizY(1:end-1),patch_size,overlap);
n_patches = length(patches);    % Delete the patches that are totally dark.
index_dark_patch = false(n_patches,1);
for i = 1:n_patches
    patch_idx = patch_to_indices(patches{i});
    Yp = data.Y(patch_idx{:},:);
    if sum(Yp(:))==0
        index_dark_patch(i) = true;
    end
end
patches(index_dark_patch) = [];
n_patches = length(patches);

K = 500;                                            % number of components to be found
tau = [3,3,3];                                          % std of gaussian kernel (half size of neuron) 
p = 0;                                            % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.85;                                  % merging threshold
sizY = data.sizY;

options = CNMFSetParms(...
    'd1',sizY(1),'d2',sizY(2),'d3',sizY(3),...
    'deconv_method','constrained_foopsi',...    % neural activity deconvolution method
    'p',p,...                                   % order of calcium dynamics
    ...% 'ssub',2,...                                % spatial downsampling when processing
    ...% 'tsub',2,...                                % further temporal downsampling when processing
    'merge_thr',merge_thr,...                   % merging threshold
    'gSig',tau,... 
    'max_size_thr',1000,'min_size_thr',19,...    % max/min acceptable size for each component
    'spatial_method','regularized',...          % method for updating spatial components
    'df_prctile',50,...                         % take the median of background fluorescence to compute baseline fluorescence 
    'fr',fr/tsub,...                            % downsamples
    'space_thresh',0.35,...                     % space correlation acceptance threshold
    'min_SNR',0.5,...                           % trace SNR acceptance threshold
    'cnn_thr',0.2,...                           % cnn classifier acceptance threshold
    'nb',1,...                                  % number of background components per patch
    ...% 'gnb',3,...                                 % number of global background components
    'decay_time',0.5,...                         % length of typical transient for the indicator used
    'MinPeakDist',5, ...
    'refine_flag',false ...                     % if it costs too much time, set this flag false
    );

%% Run on patches (the main work is done here)

[A,b,C,f,S,P,RESULTS,YrA] = run_CNMF_patches(data,K,patches,tau,0,options);  % do not perform deconvolution here since
                                                                               % we are operating on downsampled data
%% compute correlation image on a small sample of the data (optional - for visualization purposes) 
%% Cn = correlation_image_max(data,6);
%% compute average image for visualization purposes
Y_Avr = mean(data.Y,4);

%% classify components

[rval_space,rval_time,max_pr,sizeA,keep] = classify_components(data,A,C,b,f,YrA,options);

% rval_space = classify_comp_corr(data,A,C,b,f,options);
% ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                                        % this test will keep processes
                                        
%% further classification with cnn_classifier
% try  % matlab 2017b or later is needed
%     [ind_cnn,value] = cnn_classifier(A,FOV,'cnn_model.h5',options.cnn_thr);
% catch
%     ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
% end     
                            
%% event exceptionality

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < options.min_fitness);

%% select components

keep = keep & ind_exc;

%% run GUI for modifying component selection (optional, close twice to save values)
% run_GUI = false;
% if run_GUI
%     Coor = plot_contours(A,Cn,options,1); close;
%     GUIout = ROI_GUI(A,options,Cn,Coor,keep,ROIvars);   
%     options = GUIout{2};
%     keep = GUIout{3};    
% end

%% view contour plots of selected and rejected components (optional)(only adapted to 2D data)
% throw = ~keep;
% Coor_k = [];
% Coor_t = [];
% figure;
%     ax1 = subplot(121); plot_contours(A(:,keep),Cn,options,0,[],Coor_k,[],1,find(keep)); title('Selected components','fontweight','bold','fontsize',14);
%     ax2 = subplot(122); plot_contours(A(:,throw),Cn,options,0,[],Coor_t,[],1,find(throw));title('Rejected components','fontweight','bold','fontsize',14);
%     linkaxes([ax1,ax2],'xy')
    
%% keep only the active components    

A_keep = A(:,keep);
C_keep = C(keep,:);

%% extract residual signals for each trace

if exist('YrA','var') 
    R_keep = YrA(keep,:); 
else
    R_keep = compute_residuals(data,A_keep,b,C_keep,f);
end
    
%% extract fluorescence on native temporal resolution

options.fr = options.fr*tsub;                   % revert to origingal frame rate
N = size(C_keep,1);                             % total number of components
T = sum(Ts);                                    % total number of timesteps
C_full = imresize(C_keep,[N,T]);                % upsample to original frame rate
R_full = imresize(R_keep,[N,T]);                % upsample to original frame rate
F_full = C_full + R_full;                       % full fluorescence
f_full = imresize(f,[size(f,1),T]);             % upsample temporal background

S_full = zeros(N,T);

P.p = 0;
ind_T = [0;cumsum(Ts(:))];
options.nb = options.gnb;
for i = 1:numFiles
    inds = ind_T(i)+1:ind_T(i+1);   % indeces of file i to be updated
    [C_full(:,inds),f_full(:,inds),~,~,R_full(:,inds)] = update_temporal_components_fast(registered_files(i).name,A_keep,b,C_full(:,inds),f_full(:,inds),P,options);
    disp(['Extracting raw fluorescence at native frame rate. File ',num2str(i),' out of ',num2str(numFiles),' finished processing.'])
end
F_full = C_full + R_full;

if p==0
    our_plot_components_3D_GUI_p0(R_full,A_keep,C_full,b,f_full,Y_Avr,options);
else
    %% extract DF/F and deconvolve DF/F traces

    [F_dff,F0] = detrend_df_f_new(A_keep,[b,ones(prod(FOV),1)],C_full,[f_full;-double(F_dark)*ones(1,T)],R_full,options);

    C_dec = zeros(N,T);         % deconvolved DF/F traces
    S_dec = zeros(N,T);         % deconvolved neural activity
    bl = zeros(N,1);            % baseline for each trace (should be close to zero since traces are DF/F)
    neuron_sn = zeros(N,1);     % noise level at each trace
    g = cell(N,1);              % discrete time constants for each trace
    if p == 1; model_ar = 'ar1'; elseif p == 2; model_ar = 'ar2'; else; error('This order of dynamics is not supported'); end

    for i = 1:N
        spkmin = options.spk_SNR*GetSn(F_dff(i,:));
        lam = choose_lambda(exp(-1/(options.fr*options.decay_time)),GetSn(F_dff(i,:)),options.lam_pr);
        [cc,spk,opts_oasis] = deconvolveCa(F_dff(i,:),model_ar,'method','thresholded','optimize_pars',true,'maxIter',20,...
                                    'window',30,'lambda',lam,'smin',spkmin);
        bl(i) = opts_oasis.b;
        C_dec(i,:) = cc(:)' + bl(i);
        S_dec(i,:) = spk(:);
        neuron_sn(i) = opts_oasis.sn;
        g{i} = opts_oasis.pars(:)';
        disp(['Performing deconvolution. Trace ',num2str(i),' out of ',num2str(N),' finished processing.'])
    end

    our_plot_components_3D_GUI(Y,A_keep,C_full+R_full,C_full,F_dff,C_dec,S_dec,b,f_full,Y_Avr,options);
end

time_total = toc(time_start); % display the time elapsed

function idx = patch_to_indices(patch)
    % helper function to build indices vector from patch start/stop indices
    idx = arrayfun(@(x,y) x:y, patch(1:2:end), patch(2:2:end), 'un', false);
end