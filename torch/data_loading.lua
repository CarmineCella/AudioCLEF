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
    fx:close ()
    fy:close ()

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
    fx:close ()
    fy:close ()

end

print ('nclasses      = ', nclasses[1][1]);
print ('train samples = ', train_sz[1][1])
print ('test samples  = ', test_sz[1][1], '\n')

-- eof
