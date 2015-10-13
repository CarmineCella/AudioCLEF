---------------------------
-- nn_classification.lua --
---------------------------

-- this file load performs a multi-class classification using a neural network;
-- results are given in term of probability distributions

require 'torch'
require 'nn'
require 'gnuplot'

print ('[neural network classification]\n')

-- load data
dofile('data_loading.lua')

-- parameters (change here)
inputs = trainset[1][1]:size()[1]
outputs = nclasses[1][1]
layers = 3
hidden = {80, 80, 60}
learningRate = 0.001
maxIteration = 1000
verbose = false
plotting = false
---------------

-- create a neural network
mlp=nn.Sequential();  -- multi-layer perceptron

print('input layer', '', 'neurons: ', inputs)
for i = 1, layers do
    print ('hidden layer ', i, 'neurons: ', hidden[i])
end
print('output layer', '', 'neurons: ', outputs)
print ('\n')

-- model
local function nonlinearity()
    return nn.Tanh()
end

prev_neurons = inputs
for i = 1, layers do
    ln = nn.Linear(prev_neurons, hidden[i])
    --ln.weight:normal(0, 0.01)
    --ln.bias:fill(0)
    mlp:add(ln)
    mlp:add(nonlinearity())
    mlp:add(nn.Dropout(0.4))

    prev_neurons = hidden[i]
end
lout = nn.Linear(prev_neurons, outputs)
mlp:add(lout)
mlp:add(nn.LogSoftMax())

--  training
print ('training the network...')

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp, criterion) --nn
trainer.learningRate = learningRate
trainer.maxIteration = maxIteration
-- trainer.shuffleIndices = false
trainer.verbose = verbose

trainer:train(trainset)
mlp:evaluate()

print('\ntesting the network on trainset...')
tr_size = train_F:size (); --test_F:size();
aux_tr=torch.Tensor(tr_size[2])
pred_tr=torch.Tensor(tr_size[1], outputs)

nSamples_tr = 0
nCorrect_tr = 0

for i = 1, tr_size[1] do
    aux_tr = train_F[i]
    pred_tr[i] = mlp:forward(aux_tr)  -- get the prediction of the mlp
    local prediction = pred_tr[i]
    value, argmax = prediction:max(1)

    nSamples_tr = nSamples_tr + 1
    if verbose == true then
        print ('prediction: ', argmax[1], 'ground truth:', trainset[i][2][1])
    end
    if argmax[1] == trainset[i][2][1] then
        nCorrect_tr = nCorrect_tr + 1
    end
end

print('** train accuracy:', nCorrect_tr/nSamples_tr*100, '% **\n')
if plotting == true then
    gnuplot.imagesc (pred_tr)
end

print('testing the network on testset...')
te_size = test_F:size();
aux_te = torch.Tensor(te_size[2])
pred_te = torch.Tensor(te_size[1], outputs)

nSamples_te = 0
nCorrect_te = 0

for i = 1, te_size[1] do
    aux_te = test_F[i]
    pred_te[i] = mlp:forward(aux_te)  -- get the prediction of the mlp
    local prediction = pred_te[i]
    value, argmax = prediction:max(1)

    nSamples_te = nSamples_te + 1

    if verbose == true then
        print ('prediction: ', argmax[1], 'ground truth:', testset[i][2][1])
    end
    if argmax[1] == testset[i][2][1] then
        nCorrect_te = nCorrect_te + 1
    end
end

print('** test accuracy:', nCorrect_te/nSamples_te*100, '% **\n')

if plotting == true then
    gnuplot.figure ()
    gnuplot.imagesc (pred_te)
end

-- eof
