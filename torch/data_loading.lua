----------------------------------------------------------------------
-- This script loads the training dataset
----------------------------------------------------------------------
require 'torch'
require 'hdf5'
require 'gnuplot'

readFile = hdf5.open('../AC_torch_batch.h5', 'r')
train_F = readFile:read('/train_F')
train_F = train_F:all()
train_labels = readFile:read('/train_labels')
train_labels = train_labels:all()

test_F = readFile:read('/test_F')
test_labels = readFile:read('/test_labels')

test_F = test_F:all()
test_labels = test_labels:all()

readFile:close()

-- gnuplot.imagesc (train_F)
-- gnuplot.figure ()
-- gnuplot.imagesc (test_F)

n = train_F:size(); -- [numInstances,dimEachInstance]
m = train_labels:size();
t = test_F:size();

dataset = {};
function dataset:size() return n[1] end -- n[1] examples

inp=torch.Tensor(n[2]);
outp=torch.Tensor(m[2]);

for i=1,dataset:size() do
    for j=1,n[2] do
        inp[j] = train_F[i][j];
    end
    for j=1,m[2] do
        outp[j] = train_labels[i][j];
    end
	dataset[i]=  {inp,outp};  --dataset[i]={input(i,:), output(i,:)}
end

-- get data set
testp=torch.Tensor(t[2]);
testset={};
function testset:size() return t[1] end -- t[1] examples

for i=1,t[1] do
    for j=1,t[2] do
        testp[j] = test_F[i][j];
    end
    testset[i] = {testp};
end

-- Now dataset has dim dataset[n[1]][2] where
-- dataset[n[1]][1]: input(n[1],:) (vector of size n[2])
-- dataset[n[1]][2]: output(m[1],:) (idem)
