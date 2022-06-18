using SuperTuples
using BenchmarkTools
using Test

macro isinferred(ex)
	quote try
				@inferred $ex
				true
			catch err
			  false
			end
	end
end


for n = [5,10,15,20,30, 32, 33, 40]
	println("-------------")
	println("n = ", n)
	
	#m = ((rand(n) .> 0.5)...,)
	m = ntuple(i-> (i==n) ? true : false, n)
	# print("findall(b)           ")
	# isinferred = @isinferred findall(m)
	# print(isinferred, "    ")
	# @btime findall($m)

	#  print("findall_(b)           ")
	# isinferred = @isinferred SuperTuples.findall_(m)
	# print(isinferred, "    ")
	# @btime SuperTuples.findall_($m)

 	# print("findall_Val(m))   ")
	# isinferred = @isinferred SuperTuples.findall_(Val(m))
	# print(isinferred, "    ")
	# @btime SuperTuples.findall_(Val($m))

	t = ntuple(i->10*i, n)
	print("t[m]           ")
	isinferred = @isinferred getindex(t, m)
	print(isinferred, "    ")
	@btime getindex($t, $m)

 	print("select(t, m)   ")
	isinferred = @isinferred SuperTuples.select(t, m)
	print(isinferred, "    ")
	@btime SuperTuples.select($t, $m)


	print("select(t, Val(m))   ")
	isinferred = @isinferred SuperTuples.select(t, Val(m))
	print(isinferred, "    ")
	@btime SuperTuples.select($t, Val($m))

end


function f()
	if @generated
		quote
			true
		end
	else
		false
	end
end

expr = quote
	function g()
	if @generated
		quote
			true
		end
	else
		false
	end
end
end
# Conclusions:
#  When m is all true, select is better than t[m] for n <= 32
#  When m is mostly false, t[m] is better for all n