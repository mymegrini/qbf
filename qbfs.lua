require 'torch'
require 'nn'
require 'optim'

local type = "qbf "

-- Reading file
file = torch.DiskFile(arg[1], 'r'):readString("*a")

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
-- print(formula)

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

uni = init(variables)
exi = init(variables)

-- print(uni)
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
		  print(f)
		  return s, v, 1
	       end
	    else
	       f[cl][v] = 0
	       f[cl][variables + 1] = f[cl][variables + 1] - 1
	       if f[cl][variables + 1] == 0
	       then
		  print(f)
		  return s, v, 0
	       end	    
	    end
	 end
      end
   end
end
-- print(result(session()))

