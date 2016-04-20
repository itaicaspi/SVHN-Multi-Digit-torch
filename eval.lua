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
cmd:text('SVHN evaluation')
cmd:text()
cmd:text('Options:')
cmd:option('-load', 'results/drop.net', 'load saved model as starting point')
cmd:option('-image', '', 'A 1x54x54 image to be classified')
cmd:option('-type', 'cuda', 'use cuda')
cmd:option('-score', false, 'score patch => crop the number')
cmd:text()
opt = cmd:parse(arg or {})

local lengthClasses = 7
local maxDigits = lengthClasses - 2 -- class 7 is "> maxDigits"
local digitClasses = 10

local model = torch.load(opt.load)

model:evaluate()

local input = image.load(opt.image)
if opt.score then
  input = input[{{},{5,30},{74,110}}]
  input = image.scale(input, 54, 54)
  input = nn.Reshape(1,54,54):forward(input)
end
image.save('/home/itaic/TermiNet/score_parsing/SVHN/data/test.png', input[1])
input:add(-input:mean())
input = nn.Reshape(1,54,54):forward(input)
input = input:cuda()
local output = model:forward(input)


-- get number with max probability
local prob = torch.Tensor(1, lengthClasses) -- the probability for the number and length  
local number = torch.Tensor(maxDigits):zero()
local numberProb = 1

-- get actual predicted sequence

for i = 1, lengthClasses do
  local maxProbs, maxIdx
  if i > 1 then
    if i > maxDigits + 1 then -- the last class is for "> maxDigits"
      maxProbs = 1
      maxIdx = 1
    else
      -- the digit with the highest probability
      local _maxProbs, _maxIdx = torch.max(output[i], 2)
      maxProbs = _maxProbs[1][1]
      maxIdx = _maxIdx[1][1]
      number[i-1] = maxIdx
      
    end
    numberProb = numberProb * maxProbs
    --prob[1][i] = output[1][1][i] + numberProb
  end
end

local _maxProbs, _maxIdx = torch.max(output[1], 2)
local result = 0
for i = 1, _maxIdx[1][1]-1 do
  result = result * 10 + number[i]
end

print(result)