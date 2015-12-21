---------------------------
-- nn_classification.lua --
---------------------------

-- this file load performs a multi-class classification using a neural network;
-- results are given in term of probability distributions

require 'torch'
require 'nn'
require 'mattorch'
--require 'gnuplot'

-- parameters (change here)
layers = 2
hidden = {80, 80, 80}
learningRate = 0.001
maxIteration = 2000
verbose = true
---------------

print ('[neural network classification]\n')

-- load data
dofile('data_loading.lua')

tr_samples = trainset:size()
te_samples = testset:size()

inputs = trainset[1][1]:size()[2]
outputs = nclasses[1][1]

print('input layer', '', 'neurons: ', inputs)
for i = 1, layers do
    print ('hidden layer ', i, 'neurons: ', hidden[i])
end
print('output layer', '', 'neurons: ', outputs)
print ('\n')

-- model
dofile ('model.lua')

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

aux_tr=torch.Tensor(inputs)
pred_tr=torch.Tensor(tr_samples, outputs)

nSamples_tr = 0
nCorrect_tr = 0

for i = 1, tr_samples do
    aux_tr = trainset[i][1]
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

print('\ntesting the network on testset...')
aux_te = torch.Tensor(inputs)
pred_te = torch.Tensor(te_samples, outputs)

nSamples_te = 0
nCorrect_te = 0

for i = 1, te_samples do
    aux_te = testset[i][1]
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

print('\n** train accuracy:', nCorrect_tr/nSamples_tr*100, '% **')
print('** test accuracy :', nCorrect_te/nSamples_te*100, '% **\n')

-- exporting convolution filters; NB: position is dependent on the model!
l1 = mlp:get (2) -- first conv layer
l2 = mlp:get (6) -- second conv layer
--l3 = mlp:get (9) -- third conv layer

w1 = l1.weight
w2 = l2.weight
--w3 = l3.weight

mattorch.save ('l1_weights.mat', w1)
mattorch.save ('l2_weights.mat', w2)
--mattorch.save ('l3_weights.mat', w3)

-- eof
