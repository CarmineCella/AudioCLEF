-- model.lua

require 'nn'


mlp=nn.Sequential();  -- multi-layer perceptron

local function nonlinearity()
    return nn.ReLU()
end

mlp:add(nn.Dropout(0.4)) -- 250
mlp:add (nn.TemporalConvolution (inputs, inputs, 5))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (3)) -- 246/3

mlp:add(nn.Dropout(0.3)) -- 82
mlp:add (nn.TemporalConvolution (inputs, inputs, 5))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (7)) -- 77 / 7

mlp:add(nn.Dropout(0.2)) -- 11
mlp:add (nn.TemporalConvolution (inputs, inputs, 3))
mlp:add (nn.ReLU(true))
mlp:add (nn.TemporalMaxPooling (9))

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
