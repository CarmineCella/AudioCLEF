-- model.lua

require 'nn'


mlp=nn.Sequential();  -- multi-layer perceptron

local function nonlinearity()
    return nn.ReLU()
end

mlp:add(nn.Dropout(0.5)) -- 250
mlp:add (nn.TemporalConvolution (inputs, inputs, 41))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (5)) -- 210

mlp:add(nn.Dropout(0.3)) -- 42
mlp:add (nn.TemporalConvolution (inputs, inputs, 15))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (3))

mlp:add(nn.Dropout(0.1)) -- 9
mlp:add (nn.TemporalConvolution (inputs, inputs, 3))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (7))

mlp:add (nn.View (inputs))


mlp:add (nn.View (inputs))

prev_neurons = inputs
for i = 1, layers do
    mlp:add(nn.Dropout(0.1))
    ln = nn.Linear(prev_neurons, hidden[i])
    --ln.weight:normal(0, 0.01)
    --ln.bias:fill(0)
    mlp:add(ln)
    mlp:add(nonlinearity())
    prev_neurons = hidden[i]
end
lout = nn.Linear(prev_neurons, outputs)
mlp:add(lout)

mlp:add(nn.LogSoftMax())


-- eof
