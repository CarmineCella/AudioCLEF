-- model.lua

require 'nn'
if useCuda == true then
  require 'cunn'
end

mlp=nn.Sequential();  -- multi-layer perceptron

if useCuda == true then
    mlp:cuda ()
end

local function nonlinearity()
    return nn.ReLU()
end

if addSpatialProcessing == true then
    mlp:add (nn.SpatialBatchNormalization(inputs, 1e-7))
    mlp:add (nn.ReLU ())
    mlp:add (nn.SpatialAdaptiveMaxPooling (1, 1))
end

mlp:add (nn.View (inputs))
prev_neurons = inputs
for i = 1, layers do
    ln = nn.Linear(prev_neurons, hidden[i])
    --ln.weight:normal(0, 0.01)
    --ln.bias:fill(0)
    mlp:add(ln)
    mlp:add(nonlinearity())
    mlp:add(nn.Dropout(0.1))
    prev_neurons = hidden[i]
end
lout = nn.Linear(prev_neurons, outputs)
mlp:add(lout)
mlp:add(nn.LogSoftMax())


-- eof
