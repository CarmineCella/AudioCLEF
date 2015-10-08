-- load the data in 'trainY.h5' and train a nn with it --

require 'torch'
require 'nn'
require 'gnuplot'

-- load a dataset --
dofile('data_loading.lua')

-- Now create a neural network
mlp=nn.Sequential();  -- make a multi-layer perceptron

-- feed-forward neural network with one hidden layer with HUs hidden units is as follows:  --

inputs =dataset[1][1]:size()[1];
print(inputs)
outputs=dataset[1][2]:size()[1];
print(outputs)

HU1s=800; -- def dim and number of hidden units (HU) --
HU2s=400; -- def dim and number of hidden units (HU) --
HU3s=120; -- def dim and number of hidden units (HU) --

mlp:add(nn.Linear(inputs,HU1s))
mlp:add(nn.Tanh())  --  we can put also the sigmoid  --
mlp:add(nn.Linear(HU1s,HU2s))
mlp:add(nn.Tanh())  --  we can put also the sigmoid  --
mlp:add(nn.Linear(HU2s,HU3s))
mlp:add(nn.Tanh())  --  we can put also the sigmoid  --
mlp:add(nn.Linear(HU3s,outputs))

--  Training a neural network --

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(mlp, criterion) --nn
trainer.learningRate = 0.0001
-- trainer.maxIteration = 0
-- trainer.shuffleIndices = false

trainer:train(dataset)

print('\n\nNow we are going to test the network')

-- Testing the neural network --
-- take the test vector from the h5 database
t = test_F:size();
pred=torch.Tensor(t)
aux=torch.Tensor(t[2])
pred=torch.Tensor(t[1],m[2])

for i=1,t[1] do
    aux[1]=test_F[i][1]
    aux[2]=test_F[i][2]
    pred[i]=mlp:forward(aux)  -- get the prediction of the mlp
    print (pred[i])
end

gnuplot.plot (pred)
