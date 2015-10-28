-- ----------------------
-- -- data_loading.lua --
-- ----------------------
--
-- -- this file load data from Matlab and put them in two torch tensors
-- -- called respectively trainset and testset
--
-- require 'torch'
-- require 'hdf5'
--
-- print ('reading data from Matlab...')
-- readFile = hdf5.open('../AC_torch_batch.h5', 'r')
-- train_F = readFile:read('/train_F')
-- train_F = train_F:all()
-- train_labels = readFile:read('/train_labels')
-- train_labels = train_labels:all()
--
-- test_F = readFile:read('/test_F')
-- test_labels = readFile:read('/test_labels')
--
-- test_F = test_F:all()
-- test_labels = test_labels:all()
--
-- nclasses = readFile:read('/nclasses'):all()
--
-- readFile:close()
--
-- print ('preparing trainset...')
-- n_tr = train_F:size(); -- [numInstances, dimEachInstance]
-- k_tr = train_labels:size();
--
-- trainset = {};
-- function trainset:size() return n_tr[1] end -- examples
--
-- inp_tr=torch.Tensor(n_tr[2]);
-- outp_tr=torch.Tensor(k_tr[2]);
--
-- for i = 1, trainset:size() do
--     for j = 1, n_tr[2] do
--         inp_tr[j] = train_F[i][j];
--     end
--     for j = 1, k_tr[2] do
--         outp_tr[j] = train_labels[i][j];
--     end
--
-- 	trainset[i]=  {inp_tr:clone(), outp_tr:clone()};
-- end
--
-- print ('preparing testset...\n')
-- n_te = test_F:size(); -- [numInstances, dimEachInstance]
-- k_te = test_labels:size();
--
-- testset = {};
-- function testset:size() return n_te[1] end -- examples
--
-- inp_te = torch.Tensor(n_te[2]);
-- outp_te = torch.Tensor(k_te[2]);
--
-- for i = 1, testset:size() do
--     for j = 1, n_te[2] do
--         inp_te[j] = test_F[i][j];
--     end
--     for j = 1, k_te[2] do
--         outp_te[j] = test_labels[i][j];
--     end
--
-- 	testset[i] =  {inp_te:clone(), outp_te:clone()};
-- end
--
-- print ('nclasses      = ', nclasses[1][1]);
-- print ('train samples = ', n_tr[1])
-- print ('test samples  = ', n_te[1], '\n')
--
-- -- eof

----------------------
-- data_loading.lua --
----------------------

-- this file load data from Matlab and put them in two torch tensors
-- called respectively trainset and testset

require 'torch'
require 'hdf5'

print ('reading data from Matlab...')
readFile = hdf5.open('../AC_torch_metadata.h5', 'r')
train_sz = readFile:read('/train_sz')
train_sz = train_sz:all()
test_sz = readFile:read('/test_sz')
test_sz = test_sz:all()
nclasses = readFile:read('/nclasses'):all()
readFile:close()
print ('preparing trainset...')

trainset = {};
function trainset:size() return train_sz[1][1] end -- examples

for i = 1, trainset:size() do
    fx = hdf5.open ('../torch_train/' .. i .. '_features.h5')
    inp_tr=fx:read('/features'):all ();
    sz = inp_tr:size ();
    fy = hdf5.open ('../torch_train/' .. i .. '_label.h5')
    out_tr=fy:read('/label'):all ()[1]; -- getting a single label since they are the same
	trainset[i]=  {inp_tr:view(1, train_sz[2][1], 1, sz[1]):clone(), out_tr:view (1):clone()};
end

print ('preparing testset...\n')

testset = {};
function testset:size() return test_sz[1][1] end -- examples

for i = 1, testset:size() do
    fx = hdf5.open ('../torch_test/' .. i .. '_features.h5')
    inp_te=fx:read('/features'):all ();
    sz = inp_te:size ();
    fy = hdf5.open ('../torch_test/' .. i .. '_label.h5')
    out_te=fy:read('/label'):all ()[1]; -- getting a single label since they are the same
	testset[i] =  {inp_te:view(1, test_sz[2][1], 1, sz[1]):clone(), out_te:view(1):clone()};
end

print ('nclasses      = ', nclasses[1][1]);
print ('train samples = ', train_sz[1][1])
print ('test samples  = ', test_sz[1][1], '\n')

-- eof
