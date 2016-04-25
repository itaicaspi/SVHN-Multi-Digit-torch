require 'nn'
require 'optim'
require 'torch'
require 'xlua'
require 'dp'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'image'
require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-save', 'results/model.net', 'subdirectory to save/log experiments in')
cmd:option('-load', '', 'load saved model as starting point')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | NAG')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 50, 'mini-batch size (1 = pure stochastic)')
cmd:option('-epochs', 500, 'number of epochs')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)') 
cmd:option('-type', 'cuda', 'use cuda')
cmd:option('-transfer', 'relu', 'activation function, options are: elu, relu')
cmd:option('-train', true, 'train the model')
cmd:option('-test', true, 'test the model')
cmd:text()
opt = cmd:parse(arg or {})

local lengthClasses = 7
local maxDigits = lengthClasses - 2 -- class 7 is "> maxDigits"
local digitClasses = 10

local function convLayer(nInput, nOutput, stride)
	local kW = 5
	local kH = 5
	local padW = (kW - 1)/2
	local padH = (kH - 1)/2
	local layer = nn.Sequential()
	layer:add(nn.SpatialConvolution(nInput, nOutput, kW, kH, 1, 1, padW, padH))
	layer:add(nn.SpatialMaxPooling(2, 2, stride, stride, 1, 1))
	layer:add(nn.SpatialBatchNormalization(nOutput))
	layer:add(nn.ReLU())
	layer:add(nn.Dropout(0.2))
	if opt.type == 'cuda' then
    return layer:cuda()
  else
    return layer
  end
end

--[[EXTRACTOR - extracts a feature vector H ]]--
extractor = nn.Sequential()
extractor:add(convLayer(1, 48, 2))
extractor:add(convLayer(48, 64, 1))
extractor:add(convLayer(64, 128, 2))
extractor:add(convLayer(128, 160, 1))
extractor:add(convLayer(160, 192, 2))
extractor:add(convLayer(192, 192, 1))
extractor:add(convLayer(192, 192, 2))
extractor:add(convLayer(192, 192, 1))
extractor:add(nn.View(192*7*7):setNumInputDims(3))
extractor:add(nn.Linear(192*7*7, 3072))
if opt.transfer == 'elu' then
  extractor:add(nn.ELU())
elseif opt.transfer == 'relu' then
  extractor:add(nn.ReLU())
end
extractor:add(nn.Linear(3072, 4096)) -- H

--[[CLASSIFIER - classifies one digit ]]--
local function classifier(classes)
	local classifier = nn.Sequential()
	classifier:add(nn.Linear(4096, classes))
	classifier:add(nn.LogSoftMax())
	if opt.type == 'cuda' then
    return classifier:cuda()
  else
    return classifier
  end
end

--[[SEQUENCER - classifies the length and all digits ]]--
sequencer = nn.ConcatTable()
sequencer:add(lengthPredictor)
sequencer:add(classifier(lengthClasses)) -- length predictor
for i = 1, maxDigits do
	sequencer:add(classifier(digitClasses)) -- digit class predictor
end

--[[MODEL]]--
model = nn.Sequential()
model:add(extractor) -- H
model:add(sequencer)

--[[LOSS]]--
criterion = nn.ClassNLLCriterion()
			

--[[ OPTIMIZATION ]]--
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = 1e-7,
  nesterov = false,
  dampening = 0
}
optimMethod = optim.sgd


-- loading datasets
trainSetPath = 'train.t7'
trainSet = torch.checkpoint(trainSetPath) 
trainSize = trainSet.labels:size()[1]
testSetPath = 'test.t7'
testSet = torch.checkpoint(testSetPath) 
testSize = testSet.labels:size()[1]

-- subtract the mean of each image
for i = 1, trainSize do
  local mean = trainSet.data[i]:mean()
  trainSet.data[i]:add(-mean)
end
for i = 1, testSize do
  local mean = testSet.data[i]:mean()
  testSet.data[i]:add(-mean)
end

-- set cuda
if opt.type == 'cuda' then
  extractor:cuda()
  sequencer:cuda()
  model:cuda()
  criterion:cuda()
end

-- initialize params
parameters,gradParameters = model:getParameters()
parameters:uniform(-0.1,0.1)	

if opt.load ~= '' then
  model = torch.load(opt.load)
end  

TrainError = 0

local function train()
	-- epochs tracking
	epoch = epoch or 1

	model:training() -- for dropout
	model:zeroGradParameters()
	
  local confusion = optim.ConfusionMatrix(10)
  
	shuffle = torch.randperm(trainSize)
	for t = 1, trainSize - opt.batchSize, opt.batchSize do
		-- display progress
		xlua.progress(t, trainSize)
		
		-- get batch
		local inputs = trainSet.data:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())
		local targets = trainSet.labels:index(1, shuffle:sub(t, t + opt.batchSize - 1):long()):transpose(1,2)  
		
    -- get sequences lengths (actually length+1)
    local _, length = torch.max(targets, 1)
    length = length[1]		
    length[length:gt(maxDigits)] = maxDigits
    
    --image.save('/home/itaic/Documents/test.png', inputs[1])
    --print(targets[{{},{1}}])
    
    if opt.type == 'cuda' then
      inputs = inputs:cuda()
      targets = targets:cuda()
    end
    
    -- evaluation function for optim
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end
      
      -- reset gradients
			gradParameters:zero()
			
			-- batch error accumulator
			local f = 0
			
			-- forward
      local output = model:forward(inputs)
			
      -- get error from criterion for length net
			local gradInput = {}
      if opt.type == 'cuda' then
        gradInput[1] = torch.Tensor(opt.batchSize, lengthClasses):zero():cuda()
        for i = 1,maxDigits do
          gradInput[i+1] = torch.Tensor(opt.batchSize, 10):zero():cuda()
        end
      else
        gradInput[1] = torch.Tensor(opt.batchSize, lengthClasses):zero()
        for i = 1,maxDigits do
          gradInput[i+1] = torch.Tensor(opt.batchSize, 10):zero()
        end
      end
      
      --print(targets)
      for b = 1, opt.batchSize do
        -- get gradients for the length tower
				local err = criterion:forward(output[1][b], length[b])
        gradInput[1][b] = criterion:backward(output[1][b], length[b])    
        f = f + err
        -- get gradients for each one of the digit towers
				for i = 1, length[b]-1 do
          local err = criterion:forward(output[i+1][b], targets[i][b])
          confusion:add(output[i+1][b], targets[i][b])
          gradInput[i+1][b] = criterion:backward(output[i+1][b], targets[i][b])
          f = f + err
        end
			end
			model:backward(inputs, gradInput)
      
      -- average gradients and error
			gradParameters:div(opt.batchSize)
			f = f / opt.batchSize
			
      TrainError = f
      
      
      --print(confusion)
      --print('Train Error = ' .. TrainError .. '\n')  
      return f, gradParameters
		end
		
		optimMethod(feval, parameters, optimState)
    
    collectgarbage('collect')

	end
	print(confusion)
  print('Train Error = ' .. TrainError .. '\n')  
end

local function test()
  model:evaluate()
  
  local confusion = optim.ConfusionMatrix(10)
  
	shuffle = torch.randperm(testSize)
	for t = 1, testSize - opt.batchSize, opt.batchSize do
		-- display progress
		xlua.progress(t, testSize)
		
		-- get batch
		local inputs = testSet.data:index(1, shuffle:sub(t, t + opt.batchSize - 1):long())
		local targets = testSet.labels:index(1, shuffle:sub(t, t + opt.batchSize - 1):long()):transpose(1,2)  
		
    -- get sequences lengths (actually length+1)
    local _, length = torch.max(targets, 1)
    length = length[1]		
    length[length:gt(maxDigits)]=maxDigits
    --image.save('/home/itaic/Documents/test.png', inputs[1])
    --print(targets[{{},{1}}])
    
    if opt.type == 'cuda' then
      inputs = inputs:cuda()
      targets = targets:cuda()
    end
    
    -- forward
    local output = model:forward(inputs)
			
    for b = 1, opt.batchSize do
      for i = 1, length[b]-1 do
        confusion:add(output[i+1][b], targets[i][b])
      end
    end  
  end
  print(confusion)
end

for e = 1, 1 do --opt.epochs do
  if opt.train then train() end
  
  -- save/log current net
  if opt.save ~= '' then
  local filename = opt.save--paths.concat(opt.save, 'model.net')
  os.execute('mkdir -p ' .. sys.dirname(filename))
  print('==> saving model to '..filename)
  torch.save(filename, model)
  end

  if opt.test then test() end
end