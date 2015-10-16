
-- model_edouard.lua

require 'nn'
if useCuda == true then
  require 'cunn'
end

mlp=nn.Sequential()  -- multi-layer perceptron
--mlp:add(nn.View (1, k, t, 1))
-- mlp:add(nn.SpatialBatchNormalization (k, 1e-5))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialAveragePooling(1,1))
mlp:add(nn.Linear(k, outputs)
mlp:add(nn.LogSoftMax())

if useCuda == true then
    mlp:cuda ()
end

-- eof
