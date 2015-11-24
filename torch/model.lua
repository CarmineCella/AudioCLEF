-- model.lua

require 'nn'


mlp=nn.Sequential();  -- multi-layer perceptron

local function nonlinearity()
    return nn.ReLU()
end

mlp:add(nn.Dropout(0.4)) -- 50
mlp:add (nn.TemporalConvolution (inputs, inputs*2, 5))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (2)) -- 46/2

mlp:add(nn.Dropout(0.3)) -- 23
mlp:add (nn.TemporalConvolution (inputs*2, inputs*4, 5))
mlp:add (nn.ReLU(true))

mlp:add(nn.Dropout(0.2)) -- 19
mlp:add (nn.TemporalConvolution (inputs*4, inputs*8, 3))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (17)) -- 17

mlp:add (nn.View (inputs*8))

prev_neurons = inputs*8
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


-- temporary backup of a working model...
--
-- require 'nn'
--
--
-- mlp=nn.Sequential();  -- multi-layer perceptron
--
-- local function nonlinearity()
--     return nn.ReLU()
-- end
--
-- mlp:add(nn.Dropout(0.2))
-- mlp:add (nn.TemporalConvolution (inputs, inputs, 10))
-- mlp:add (nn.ReLU(true))
--
-- --
-- -- mlp:add(nn.Dropout(0.2))
-- -- mlp:add (nn.TemporalConvolution (inputs/2, inputs/4, 1))
-- -- mlp:add (nn.ReLU(true))
-- -- mlp:add (nn.TemporalMaxPooling (25))
--
-- mlp:add (nn.View (inputs))
--
-- prev_neurons = inputs
-- for i = 1, layers do
--     mlp:add(nn.Dropout(0.2))
--     ln = nn.Linear(prev_neurons, hidden[i])
--     --ln.weight:normal(0, 0.01)
--     --ln.bias:fill(0)
--     mlp:add(ln)
--     mlp:add(nonlinearity())
--     prev_neurons = hidden[i]
-- end
-- lout = nn.Linear(prev_neurons, outputs)
-- mlp:add(lout)
-- mlp:add (nn.TemporalMaxPooling (241))
-- mlp:add(nn.LogSoftMax())

-- eof
