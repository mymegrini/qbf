require 'torch'
require 'nn'
require 'optim'

local type = "qbf "

-- Reading file
file = torch.DiskFile(arg[1], 'r'):readString("*a")
print("File :")
print(file)

-- Checking type
if file:find(type) ~= 1
then
   print("File type unrecognized")
   return
end

-- Parsing variable and clause counts
index = type:len() + 1
list = file:gmatch("[-]?%d+")

variables = tonumber(list())
clauses = tonumber(list())

formula = torch.Tensor(clauses, variables + 1):zero()

-- Parsing clauses
local cl = 1
local v = 0

while cl <= clauses do

   v = tonumber(list())

   if v < -variables or v > variables
   then
      print(string.format("Incoherent values. %d [%d]", v, variables))
      return
   elseif v == 0
   then
      cl = cl + 1
   elseif v > 0
   then
      formula[cl][v] = 1
      formula[cl][variables + 1] = formula[cl][variables + 1] + 1
   else
      formula[cl][-v] = -1
      formula[cl][variables + 1] = formula[cl][variables + 1] + 1
   end
end
print("Formula:")
print(formula)

-- Creating a neural network
function init(size)
   local net = nn.Sequential()
   local hidden = size
   local output = 1
   net:add(nn.Linear(variables, hidden))
   net:add(nn.Tanh())
   net:add(nn.Linear(hidden, output))
   net:add(nn.Tanh())
   return net
end

uni = init(variables) -- aims for loss
exi = init(variables) -- aims for win

print("Neural network model:")
print(uni)
-- print(uni(torch.Tensor(variables):fill(0)))
-- print(exi(torch.Tensor(variables):fill(0)))

-- Making random choice
function play(output)
   local v = 2 * torch.uniform() -1 + output
   if v < 0
   then
      return -1
   else
      return 1
   end
end
print("")

-- showing progression
score = 0
count = 0
function percentage(i)
   if i == 0 then
      --print(string.format("%d%%", math.floor(100*score/count)))
      io.write(string.format("\r\r\r\r\r%d%% ", math.floor(100*score/count)))
   else
      score = score + (i+1)/2
      count = count + 1
   end
end

-- Running an evaluation session
function session()
   local s = torch.Tensor(2, variables):fill(0)
   for i = 1,variables do
      if i % 2 == 0
      then
	 s[2][i] = exi(s[1])
      else
	 s[2][i] = uni(s[1])
      end
      s[1][i] = play(s[2][i])
   end
   return s
end
-- print(session())

-- Determining result
function result(s)
   local f = formula:clone()
   local count = clauses
   for v = 1,variables do
      for cl = 1,(f:size()[1]) do
	 if f[cl][v] ~= 0
	 then
	    if f[cl][v] == s[1][v]
	    then
	       f[cl]:zero()
	       count = count - 1
	       if count == 0
	       then
		  percentage(1)
		  return s, v, 1
	       end
	    else
	       f[cl][v] = 0
	       f[cl][variables + 1] = f[cl][variables + 1] - 1
	       if f[cl][variables + 1] == 0
	       then
		  percentage(-1)
		  return s, v, -1
	       end
	    end
	 end
      end
   end
end
-- print(result(session()))

-- Building training data sets
function build(s, v, r)
   print(s, v, r)
   local lambda = .5
   local u = math.floor((variables + 1)/2)
   local e = math.floor(variables/2)
   local uni_set = torch.Tensor(u, variables):zero()
   local exi_set = torch.Tensor(e, variables):zero()
   local uni_val = torch.Tensor(u)
   local exi_val = torch.Tensor(e)

   local function pull(value, target, factor, i)
      print(value, target, factor, i)
      if (i>v) then return value
      else return value * (1-factor) + target * factor
      end
   end

   for i=1,v do
      local f = lambda ^ math.max(v-i,0)
      if i % 2 == 0
      then
	 for j=i/2,e-1 do
	    exi_set[j+1][i] = s[1][i]
	    if j<u then uni_set[j+1][i] = s[1][i] end
	 end
	 exi_val[i/2] = pull(s[2][i], s[1][i], f, i) -- incentivizing
      else
	 for j=(i-1)/2+1,u do
	    if j<u then uni_set[j+1][i] = s[1][i] end
	    exi_set[j][i] = s[1][i]
	 end
	 uni_val[(i-1)/2+1] = pull(s[2][i], -s[1][i], f, i) -- disincentivizing
      end
   end
   return uni_set, uni_val, exi_set, exi_val
end

-- Training models
function train(model, input, target)

   local criterion = nn.MSECriterion()
   local x, dl_dx = model:getParameters()

   local function eval(_x)
      if _x ~= x then
	 x:copy(_x)
      end

      local sample = (sample or 0) + 1
      if sample > (#target)[1] then sample = 1 end

      dl_dx:zero()

      local loss =
	 criterion:forward(model:forward(input[sample]), target[{{sample}}])

      model:backward(input[sample],
		     criterion:backward(model.output, target[{{sample}}]))

      return loss, dl_dx
   end

   sgd_params = {
      learningRate = .1,
      learningRateDecay = .0001,
      weightDecay = 0,
      momentum = 0
   }

   for i = 1,100*variables do
      for i =1,(#target)[1] do
	 _,fs = optim.sgd(eval, x, sgd_params)
      end
   end
end

print("Running:")

-- running the algorithm
n = 1
while n>0 do
   us, uv, es, ev = build(result(session()))
   train(uni, us, uv)
   train(exi, es, ev)
   percentage(0)
   n = n-1
end
print("")
