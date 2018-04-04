% modify this according to your own log
rain_log_file = '/work/dev/experiments/py-mask-rcnn/experiments/logs/faster_rcnn_end2end_VGG_CNN_M_1024_.txt.2017-12-16_10-22-13';
train_interval = 20;
test_interval = 20;

[~,string_output] = dos(['cat ', train_log_file, ' | grep ", loss = " | awk ''{print $13}''']);
train_loss = str2num(string_output);
n = 1:length(train_loss);
idx_train = (n-1) * train_interval;

figure;plot(idx_train, train_loss);
hold on;

grid on;
legend('Train Loss', 'Test Loss');
xlabel('iterations');
ylabel('loss');
title(' Train Loss Curve');