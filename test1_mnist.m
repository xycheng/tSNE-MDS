
clear all;
rng(3172017)

%% load mnist 5000
load MNIST_X_te_all_32.mat

nperclass=200;
Nclass=10;

data=zeros(nperclass*Nclass,28^2);
labels=zeros(nperclass*Nclass,1);
for iclass=1:Nclass
    id=(iclass-1)*nperclass+1:iclass*nperclass;
    x=X_te{iclass}(:,1:nperclass)';
    x= reshape(x,nperclass,32,32);
    x=x(:,3:30,3:30);
    data(id,:)=reshape(x,[nperclass,28^2])/255;
    labels(id)=iclass;
end
n=nperclass*Nclass;


% rescale data so that every sample is in unit ball in high dim
nx=sqrt(sum(data.^2,2));
data=data/max(nx);

%% Barnes-Hut tsne
addpath ./bhtsne/

no_dims=2;
initial_dims=50;
perplexity=30;
theta=0.5;
mappedX = fast_tsne(data,no_dims,initial_dims,perplexity,theta);

%%

figure(11),clf;
scatter(mappedX(:,1),mappedX(:,2),40,labels,'o','filled');
colormap(jet);
