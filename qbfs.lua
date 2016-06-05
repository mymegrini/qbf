require 'torch'
require 'nn'
require 'optim'

local halfline = "----------------------------------------"
print("\nNNQBF\n")

-- Choosing number of epochs, seconds and verbosity
if #arg >= 2 -- epochs
then
   n = tonumber(arg[2])
else
   n = 1000
end
if #arg >= 3 -- print delay
then
   step = tonumber(arg[3])
else
   step = 1
end
if #arg >= 4 -- verbosity
then
   verbosity = tonumber(arg[4])
else
   verbosity = 2
end

-- Reading file
file = torch.DiskFile(arg[1], 'r'):readString("*a")
if verbosity>=2
then
   print("File :")
   print(file)
end

-- Checking type
local type = "p cnf "
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
      if verbosity >= 4
      then
	 print("sets:")
	 print(s)
	 print(a, e)
      end
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

   if verbosity>=3
   then
      print("Reindexing:")
      print(atomindex)
      print(quantindex)
   end

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

quantifiers(file)
-- Counting atoms for each quantifier
u = 0
e = 0
for i=1,variables do
   e = e + quantifier[1][i]
   u = u + (1-quantifier[1][i])
end
if verbosity>=2 then
   print("Reindexing:")
   print(index)
   print("Quantifiers:")
   print(quantifier)
end

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
if verbosity>=2 then
   print("Formula:")
   print(formula)
end

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

if verbosity>=2 then
   print("Neural network model:")
   print(uni)
end

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
   -- batch sizes
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

-- inputs memory
su = math.max(u,10)*u
se = math.max(e,10)*e
uni_mset = torch.Tensor(su, variables):zero()
exi_mset = torch.Tensor(se, variables):zero()
-- targets memory
uni_mval = torch.Tensor(su):zero()
exi_mval = torch.Tensor(se):zero()
-- data sizes
uni_msize = 0
exi_msize = 0
-- Storing data into memory
function store(us, uv, ui, es, ev, ei)
   for i=1,ui do
      -- check for duplicates
      for j=1,uni_msize do
	 if torch.all(torch.eq(uni_mset[j], us[ui]))
	 then
	    uni_mval[j] = uv[ui]
	    goto u_done
	 end
      end
      -- no duplicate found
      if uni_msize < su
      then
	 uni_msize = uni_msize + 1
	 uni_mset[uni_msize] = us[ui]
	 uni_mval[uni_msize] = uv[ui]
      else
	 local k = torch.random(1, su)
	 uni_mset[k] = us[ui]
	 uni_mval[k] = uv[ui]
      end
      ::u_done::
   end
   for i=1,ei do
      -- check for duplicates
      for j=1,exi_msize do
	 if torch.all(torch.eq(exi_mset[j], es[ei]))
	 then
	    exi_mval[j] = ev[ei]
	    goto e_done
	 end
      end
      -- no duplicate found
      if exi_msize < se
      then
	 exi_msize = exi_msize + 1
	 exi_mset[exi_msize] = es[ei]
	 exi_mval[exi_msize] = ev[ei]
      else
	 local k = torch.random(1, se)
	 exi_mset[k] = es[ei]
	 exi_mval[k] = ev[ei]
      end
      ::e_done::
   end
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

	 local sample = torch.random(1, size)

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
      
      for i =1,(10*(#target)[1]) do
	 _,fs = optim.sgd(eval, x, sgd_params)
	 l = l + fs[1]
      end
      l = l / (10*(#target)[1])
      return l
   else
      return 0
   end
end

function line()
   print(halfline .. halfline .. "\n")
end

if verbosity>=1 then
   print("Press Enter to start.")
   io.stdin:read('*l')
   print("Running:\n")
   line()
end

-- running the algorithm
game = 0 -- current number of game sessions

t = os.clock()
while n>0 do
   local s, v, r = result(session())
   local us, uv, ui, es, ev, ei = build(s, v, r)
   store(us, uv, ui, es, ev, ei)
   local ul = train(uni, uni_mset, uni_mval, uni_msize)
   local el = train(exi, exi_mset, exi_mval, exi_msize)
   
   if game == 0 or os.clock() > t + step
   then
      if game>0 then t = os.clock() end
      if verbosity>=1 then
	 line()
	 print("Session " .. game ..":")
	 print(s)
	 print("Result: " .. ((r+1)/2) .. " (var " .. v .. ")\n")
      end
      if verbosity>=2 then
	 print("Training set:\n")
	 print("uni:")
	 print(us, uv)
	 print("exi:")
	 print(es, ev)
      end
      if verbosity>=3 then
	 print("Training data:\n")
	 print("uni:")
	 print(uni_mset, uni_mval)
	 print("exi:")
	 print(exi_mset, exi_mval)
      end
      if verbosity>=1 then
	 print("\nLoss: uni" .. ul .. " exi" .. el .."\n")
	 print("\nPercentage:")
	 percentage(-1)
      end
   end
   game = game + 1
   n = n-1
end

-- print final session without exploration
explore = false
local s, v, r = result(session())
if verbosity>=1 then
   line()
   print("Session " .. game ..":")
   print(s)
   print("Result: " .. ((r+1)/2) .. " (var " .. v .. ")")
   print("\nPercentage:")
end
percentage(-1)