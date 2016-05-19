require 'torch'
require 'nn'
require 'optim'

local type = "p cnf "
print("\nNNQBF\n")

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
quantifier = torch.Tensor(variables):zero()

u = 0
e = 0

-- Parsing quantifiers (placeholder)
for i=1,variables do
   if i % 2 == 0
   then
      quantifier[i] = 1
      e = e + 1
   end
end
u = variables - e

-- Parsing clauses
local cl = 1
local var = 0

while cl <= clauses do

   var = tonumber(list())

   if var < -variables or var > variables
   then
      print(string.format("Incoherent values. %d [%d]", var, variables))
      return
   elseif var == 0
   then
      cl = cl + 1
   elseif var > 0
   then
      formula[cl][var] = 1
      formula[cl][variables + 1] = formula[cl][variables + 1] + 1
   else
      formula[cl][-var] = -1
      formula[cl][variables + 1] = formula[cl][variables + 1] + 1
   end
end
print("Quantifier:")
print(quantifier)
print("Formula:")
print(formula)

-- Creating a neural network
function init(size)
   local net = nn.Sequential()
   local hidden = size
   local output = 1
   net:add(nn.Linear(variables, hidden))
   net:add(nn.Sigmoid())
   net:add(nn.Linear(hidden, output))
   net:add(nn.Sigmoid())
   return net
end

uni = init(variables) -- aims for loss
exi = init(variables) -- aims for win

print("Neural network model:")
print(uni)
-- print(uni(torch.Tensor(variables):fill(0)))
-- print(exi(torch.Tensor(variables):fill(0)))


-- Making random choice
--[[
function play(output)
   local v = 2 * torch.uniform() -1 + output
   if v < 0
   then
      return -1
   else
      return 1
   end
end
]]
print("")

-- showing progression
score = 0
count = 0
function percentage(i)
   if i == -1 then
      print(string.format("%d%%", math.floor(100*score/count)))
      --io.write(string.format("\r\r\r\r\r%d%% ", math.floor(100*score/count)))
   else
      score = score + i
      count = count + 1
   end
end

-- Making a play
function play(model, s, i)
   s[1][i] = -1
   local vf = model(s[1])[1]
   s[1][i] = 1
   local vt = model(s[1])[1]
   if vf < vt
   then
      s[1][i] = 1
      s[2][i] = vt
   else
      s[1][i] = -1
      s[2][i] = vf
   end
end

-- Running an evaluation session
function session()
   local s = torch.Tensor(2, variables):fill(0)
   for i = 1,variables do
      if quantifier[i] == 0
      then
	 play(uni, s, i)
      else
	 play(exi, s, i)
      end
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
		  percentage(0)
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

   --print(s, v, r)

   local lambda = .5
   -- learning inputs
   local uni_set = torch.Tensor(u, variables):zero()
   local exi_set = torch.Tensor(e, variables):zero()
   -- learning targets
   local uni_val = torch.Tensor(u):zero()
   local exi_val = torch.Tensor(e):zero()
   local uni_size = 0
   local exi_size = 0

   local function shift(value, target, factor)
      return (1 - factor) * value + factor * (target + 1) / 2
   end

   local e_i = 0
   local u_i = 0

   for i=1,v do
      local f = lambda ^ math.max(v-i,0)
      if quantifier[i] == 1
      then
	 e_i = e_i + 1
	 for k=1,i do exi_set[e_i][k] = s[1][k] end
	 exi_val[e_i] = shift(s[2][i], r, f)
      else
	 u_i = u_i + 1
	 for k=1,i do uni_set[u_i][k] = s[1][k] end
	 uni_val[u_i] = shift(s[2][i], -r, f)
      end
   end
   return uni_set, uni_val, u_i, exi_set, exi_val, e_i
end

-- Training models
function train(model, input, target, size)

   if size>0
   then
      local criterion = nn.MSECriterion()
      local x, dl_dx = model:getParameters()

      local function eval(_x)
	 if _x ~= x then
	    x:copy(_x)
	 end

	 local sample = (sample or 0) + 1
	 if sample > size then sample = 1 end

	 dl_dx:zero()

	 local loss =
	    criterion:forward(model:forward(input[sample]), target[{{sample}}])

	 model:backward(input[sample],
			criterion:backward(model.output, target[{{sample}}]))

	 return loss, dl_dx
      end

      sgd_params = {
	 learningRate = .01,
	 learningRateDecay = .0001,
	 weightDecay = 0,
	 momentum = 0
      }

      for i = 1,1e3 do
	 for i =1,(#target)[1] do
	    _,fs = optim.sgd(eval, x, sgd_params)
	 end
      end
   end
end

print("Running:")

-- running the algorithm
n = 100 -- total number of sessions
game = 1 -- current number of game sessions
while n>0 do
   local s, v, r = result(session())
   local us, uv, ui, es, ev, ei = build(result(session()))
   train(uni, us, uv, ui)
   train(exi, es, ev, ei)
   if n % 10 == 0
   then
      print("Session " .. game ..":")
      print(s)
      print("Result: " .. r .. " (var " .. v .. ")")

      print("Training set: ")
      print("uni:")
      print(us, uv, ui)

      print("exi:")
      print(es, ev, ei)

      print("\nPercentage:")
      percentage(-1)
      print("------------------------------------------------------\n")
   end
   game = game + 1
   n = n-1
end
