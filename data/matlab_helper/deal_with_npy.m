pred = readNPY('output/predictions.npy');
%pred_zoomed =  readNPY('pred_zoomed.npy');
gt_depth = readNPY('gt_depth.npy');
%mask = readNPY('mask.npy');

pred_single = squeeze(pred(1,:,:));
%pred_zoomed_single = pred_zoomed(1,:,:);
gt_single = squeeze(gt_depth(1,:,:));
%mask_single = mask(1,:,:);

%setting about conversion
alpha_ = 1.0
beta_ = 80.999
K_ = 71.0
%label depth conversion
depth = 0.5*(alpha_ * (beta_ / alpha_).^(pred_single(1,:,:) / K_)+alpha_ * (beta_ / alpha_).^((pred_single(1,:,:)+1.0) / K_))-0.999;

%label depth conversion
gt_label_single =K_ * log(gt_single+0.999 / alpha_) / log(beta_ / alpha_);

%resize image
resize_pred_label = imresize(squeeze(pred_single), [375,1242], 'bilinear');

%plot
[X,Y] = meshgrid(1:416,1:128);surf(X,Y,squeeze(pred_single));

% %other checking
% gap=resize_pred_label-gt_label_single;
% index = find(abs(gap)>10);gap(index)=0;
% index_p = find(gap>0);number_p=length(index_p);
% index_n = find(gap<0);number_n=length(index_n);
% max_dev = max(gap(index_p));
% [max_dev, max_index]= max(gap(index_p));
% index_p(max_index)