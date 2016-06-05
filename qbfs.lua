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
list = file:gmatch("[-]?%d+") --looking for integers

variables = tonumber(list())
clauses = tonumber(list())

formula = torch.Tensor(clauses, variables + 1):zero()
index = torch.Tensor(1, variables):zero()
quantifier = torch.Tensor(1, variables):zero()

-- Parsing atom sets
function quantifiers(s)

   local quantindex = torch.Tensor(variables):fill(1) -- default quantifier is e
   local atomindex = torch.Tensor(variables):zero()
   -- Indexing atom sets
   local function atoms(s, i, q)

      list = s:gmatch("[-]?%d+")
      var = tonumber(list())
      while (var ~= 0) do
	 if var < 0 or var > variables
	 then
	    print(string.format("Incorrect atom value. %d [%d]",
				var, variables))
	    return
	 else
	    atomindex[var] = i
	    quantindex[var] = q
	 end
	 var = tonumber(list())
      end
      return iter
   end

   -- Finding atom sets and indexing them
   local function sets(s, i)

      local a = s:find("a ")
      local e = s:find("e ")
      --[[
      print("sets:")
      print(s)
      print(a, e)
      ]]
      if a ~= nil
      then
	 if e ~= nil
	 then
	    if a < e
	    then
	       iter = atoms(s:sub(a+2), i, 0)
	       return a + 2
	    else
	       iter = atoms(s:sub(e+2), i, 1)
	       return e + 2
	    end
	 else
	    iter = atoms(s:sub(a+2), i, 0)
	    return a+2
	 end
      else
	 if e ~= nil
	 then
	    iter = atoms(s:sub(e+2), i, 1)
	    return e + 2
	 else
	    return nil
	 end
      end
   end

   local i = 1
   local set = 1
   local s = file

   -- Parsing sets
   while set ~= nil do
      s = s:sub(set)
      set = sets(s, i)
      i = i + 1
   end

   --[[
   print("Reindexing:")
   print(atomindex)
   print(quantindex)
   ]]

   local qset = 0
   local var = 1
   while (var <= variables) do
      for i=1,variables do
	 if atomindex[i] == qset
	 then
	    index[1][i] = var
	    quantifier[1][var] = quantindex[i]
	    var = var + 1
	 end
      end
      qset = qset + 1  -- Switching to next set
   end
end

print("Reindexing:")
quantifiers(file)
print(index)
print("Quantifiers:")
-- Counting atoms for each quantifier
u = 0
e = 0
for i=1,variables do
   e = e + quantifier[1][i]
   u = u + (1-quantifier[1][i])
end
print(quantifier)

-- Parsing clauses
local cl = 1
local var = 0

while cl <= clauses do
   --print(cl)
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
      formula[cl][index[1][var]] = 1
      formula[cl][variables + 1] = formula[cl][variables + 1] + 1
   else
      formula[cl][index[1][-var]] = -1
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
   else
      score = score + i
      count = count + 1
   end
end

-- Making a play
explore = true  -- allows networks to choose random plays
function play(model, s, i)
   s[1][i] = -1
   local vf = model(s[1])[1]
   s[1][i] = 1
   local vt = model(s[1])[1]
   if explore
   then  -- random
      local b = vf + vt
      if (vf + vt) * torch.uniform() < vt
      then
	 s[1][i] = 1
	 s[2][i] = vt
      else
	 s[1][i] = -1
	 s[2][i] = vf
      end
   else  -- deterministic
      if vf < vt
      then
	 s[1][i] = 1
	 s[2][i] = vt
      else
	 s[1][i] = -1
	 s[2][i] = vf
      end
   end
end

-- Running an evaluation session
function session()
   local s = torch.Tensor(2, variables):fill(0)
   for i = 1,variables do
      if quantifier[1][i] == 0
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
      if quantifier[1][i] == 1
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
	 learningRateDecay = .001,
	 weightDecay = 0,
	 momentum = 0
      }

      local l = 0
      for i = 1,1e2 do
	 current_loss = 0
	 for i =1,(#target)[1] do
	    _,fs = optim.sgd(eval, x, sgd_params)
	    current_loss = current_loss + fs[1]
	 end
	 current_loss = current_loss / (#target)[1]
	 l = l + current_loss
      end
      return l/1e3
   end
end

print("Press Enter to start.")
io.stdin:read('*l')
print("Running:\n")
print("-------------------------------------------------------------------------------\n")

-- running the algorithm
game = 0 -- current number of game sessions
-- Choosing number of epochs
if #arg >= 2
then
   n = tonumber(arg[2])
else
   n = 1000
end
if #arg >= 3
then
   step = tonumber(argv[3])
else
   step = 1
end
t = os.clock()
while n>0 do
   local s, v, r = result(session())
   local us, uv, ui, es, ev, ei = build(result(session()))
   local ul = train(uni, us, uv, ui)
   local el = train(exi, es, ev, ei)
   if os.clock() > t + step
   then
      t = os.clock()
      print("Session " .. game ..":")
      print(s)
      print("Result: " .. ((r+1)/2) .. " (var " .. v .. ")\n")
      --[[
      print("Training set:\n")
      print("uni: " .. ul)
      print(us, uv)
      print("exi: " .. el)
      print(es, ev)
      ]]
      print("\nPercentage:")
      percentage(-1)
print("-------------------------------------------------------------------------------\n")
   end
   game = game + 1
   n = n-1
end

-- print final session without exploration
explore = false
local s, v, r = result(session())
print("Session " .. game ..":")
print(s)
print("Result: " .. ((r+1)/2) .. " (var " .. v .. ")")
print("\nPercentage:")
percentage((r+1)/2)
percentage(-1)
