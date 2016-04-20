require 'image'
require 'xlua'
require 'dp'

----------------------------------------------------------------------

function ls(path) return sys.split(sys.ls(path),'\n') end -- alf ls() nice function!

----------------------------------------------------------------------

local maxDigits = 6

function digitsVec(num) 
  local res = torch.Tensor(maxDigits+1):fill(11) -- class 10 is end-of-seq
  local num = tostring(num)
  local idx = 1
  for digit in num:gmatch('%d') do
    res[idx]=tonumber(digit)
    if res[idx] == 0 then
      res[idx] = 10
    end
    idx = idx + 1
  end
  return torch.Tensor(res)
end

---------------------------------------------------------------------

-- Data params:

local channels = 1
local height = 54
local width = 54

local endLabel = 10
local validationSize = 5000


local train_dir = 'train/'
local train_zip = './train.zip'
local train_labels = './train_labels.txt'
local train_url = "https://www.deep-ai.xyz/datasets/train.zip"

local extra_dir = 'extra/'
local extra_zip = './extra.zip'
local extra_labels = './extra_labels.txt'
local extra_url = "https://www.deep-ai.xyz/datasets/extra.zip"

local test_dir = 'test/'
local test_zip = './test.zip'
local test_labels = './test_labels.txt'
local test_url = "https://www.deep-ai.xyz/datasets/test.zip"

local trainSetPath = 'train.t7'
local validSetPath = 'validation.t7'
local testSetPath = 'test.t7'

-----------------------------------------------------------------------
if not paths.filep(trainSetPath) or not paths.filep(validSetPath) then

  if not paths.filep(train_zip) then os.execute('wget '..train_url)   end
  if not paths.filep(extra_zip) then os.execute('wget '..extra_url)   end

  print(sys.COLORS.red ..  '==> Unzipping training data')

  -- unzip training data into train dir
  if not paths.dirp(train_dir) then
     os.execute('mkdir -p ' .. train_dir)
     os.execute('unzip -q ' .. train_zip .. ' -d ' .. train_dir)
  end

  local trainFiles = ls(train_dir)
  local trainSize = #trainFiles

  print(sys.COLORS.red ..  '==> Unzipping extra training data')

  -- unzip extra training data into train dir
  if not paths.dirp(extra_dir) then
     os.execute('mkdir -p ' .. extra_dir)
     os.execute('unzip -q ' .. extra_zip .. ' -d ' .. extra_dir)
  end

  local extraFiles = ls(extra_dir)
  local extraSize = #extraFiles

  ---------------------------------------------------------------------------

  local trainData = {
    data = torch.Tensor(trainSize + extraSize - validationSize, channels, width, height),
    labels = torch.Tensor(trainSize + extraSize - validationSize, maxDigits+1),
    size = function() return (trainSize + extraSize - validationSize) end
  }

  local validationData = {
    data = torch.Tensor(validationSize, channels, width, height),
    labels = torch.Tensor(validationSize, maxDigits+1),
    size = function() return validationSize end
  }

  ----------------------------------------------------------------------------

  local dataShuffle = torch.randperm(trainSize + extraSize)

  print(sys.COLORS.red ..  '==> Loading training data to memory')

  -- Load training data into memory
  local trainLabelsFile = assert(io.open(train_labels, "rb"))
  for i = 1, trainSize do
    if dataShuffle[i] > validationSize then
      trainData.data[dataShuffle[i] - validationSize] = image.loadPNG(train_dir .. i .. '.png', channels)
      trainData.labels[dataShuffle[i] - validationSize] = digitsVec(trainLabelsFile:read())
    else
      validationData.data[dataShuffle[i]] = image.loadPNG(train_dir .. i .. '.png', channels)
      validationData.labels[dataShuffle[i]] = digitsVec(trainLabelsFile:read())
    end
    xlua.progress(i, trainSize)
  end

  -- delete extracted files
  os.execute('rm -rf ' .. train_dir)

  print(sys.COLORS.red ..  '==> Loading extra training data to memory')

  -- Load extra training data into memory
  local extraLabelsFile = assert(io.open(extra_labels, "rb"))
  for i = 1, extraSize do
    if dataShuffle[trainSize+i] > validationSize then
      trainData.data[dataShuffle[trainSize+i] - validationSize] = image.loadPNG(extra_dir .. i .. '.png', channels)
      trainData.labels[dataShuffle[trainSize+i] - validationSize] = digitsVec(extraLabelsFile:read())
    else
      validationData.data[dataShuffle[trainSize+i]] = image.loadPNG(extra_dir .. i .. '.png', channels)
      validationData.labels[dataShuffle[trainSize+i]] = digitsVec(extraLabelsFile:read())
    end
    xlua.progress(i, extraSize)
  end

  -- delete extracted files
  os.execute('rm -rf ' .. extra_dir)

  print(sys.COLORS.red ..  '==> Saving training and validation data to t7 files')

  -- Save created dataset:
  torch.save('train.t7',trainData)
  torch.save('validation.t7',validationData)

  trainData = nil
  validationData = nil
  
end

------------------------------------------------------------------------------------------

if not paths.filep(testSetPath) then

  if not paths.filep(test_zip) then os.execute('wget '..test_url)   end
  print(sys.COLORS.red ..  '==> Unzipping test data')

  -- unzip test data into test dir
  if not paths.dirp(test_dir) then
     os.execute('mkdir -p ' .. test_dir)
     os.execute('unzip -q ' .. test_zip .. ' -d ' .. test_dir)
  end

  local testSize = #ls(test_dir)

  testData = {
    data = torch.Tensor(testSize, channels, width, height),
    labels = torch.Tensor(testSize, maxDigits+1),
    size = function() return testSize end
  }

  local testShuffle = torch.randperm(testSize)

  print(sys.COLORS.red ..  '==> Loading test data to memory')

  -- Load training data into memory
  local testLabelsFile = assert(io.open(test_labels, "rb"))
  local files = ls(test_dir)
  for i = 1, testSize do
    testData.data[testShuffle[i]] = image.loadPNG(test_dir .. i .. '.png', channels)
    testData.labels[testShuffle[i]] = digitsVec(testLabelsFile:read())
    xlua.progress(i, testSize)
  end


  -- delete extracted files
  os.execute('rm -rf ' .. test_dir)

  print(sys.COLORS.red ..  '==> Saving training data to t7 file')

  -- Save created dataset:
  torch.save('test.t7',testData)
  
end
--------------------------------------------------------------------------------------------

print(sys.COLORS.green ..  '==> Success!')