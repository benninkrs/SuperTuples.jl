# Additions/extensions to Julia that I (RSB) like
module SuperTuples

export oneto, tupseq, tupseqd, filltup, invpermute, select, antiselect, findin, tcat

using StaticArrays
import Base: getindex, setindex!
import Base: ntuple, invperm, sort, sortperm, in, allunique, (+), (-), (*)
import Base.Iterators.take


# # Extract length of a tuple type
# length(::Type{T}) where {T<:NTuple{N}} where {N} = N

# Tuple constructors

"""
	ntuple(f, n, T)

Create an `NTuple{N,T}` with values `(f(1), ..., f(n))`. For n>10, this is much faster
than ntuple(f, n).  An `InexactError` will occur if any of `f(1)`,...,`f(n)` cannot be
converted to type `T`.
"""
function ntuple(f, n::Integer, ::Type{T}) where {T}
	t = n == 0  ? () :
		n == 1  ? NTuple{n,T}((f(1),)) :
		n == 2  ? NTuple{n,T}((f(1), f(2))) :
		n == 3  ? NTuple{n,T}((f(1), f(2), f(3))) :
		n == 4  ? NTuple{n,T}((f(1), f(2), f(3), f(4))) :
		n == 5  ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5))) :
		n == 6  ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6))) :
		n == 7  ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7))) :
		n == 8  ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8))) :
		n == 9  ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9))) :
		n == 10 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10))) :
		begin
			v = MVector{n, T}(undef)
			@inbounds for i in 1:n
				v[i] = f(i)
			end
			Tuple(v)
		end
end


# This is only slightly faster than Base.map for 4<n<=10, and essentially the same for n>10.
#Also, f.(t) is somehow the fastest of all (!)
# @inbounds didn't help
# function map_(f, t::Tuple, ::Type{T}) where {T}
# 	n = length(t)
# 	s = n == 0  ? () :
# 		n == 1  ? (f(t[1]),) :
# 		n == 2  ? (f(t[1]), f(t[2])) :
# 		n == 3  ? (f(t[1]), f(t[2]), f(t[3])) :
# 		n == 4  ? (f(t[1]), f(t[2]), f(t[3]), f(t[4])) :
# 		n == 5  ? (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5])) :
# 		n == 6  ? (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6])) :
# 		n == 7  ? (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7])) :
# 		n == 8  ? (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8])) :
# 		n == 9  ? (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8]), f(t[9])) :
# 		n == 10 ? (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8]), f(t[9]), f(t[10])) :
# 		begin
# 			v = MVector{n, T}(undef)
# 			i = 1
# 			while i <= n
# 				v[i] = f(t[i])
# 				i += 1
# 			end
# 			Tuple(v)
# 		end
# 	return s
# end


# This is much slower than map_ or ntuple for n>10 ... why???
# It seems silly one one even have to do this
# function take(t::Tuple, n::Integer)
# 	n > length(t) && error("n must be less than length(t)")
# 	s = n == 0  ? () :
# 		n == 1  ? (t[1],) :
# 		n == 2  ? (t[1], t[2]) :
# 		n == 3  ? (t[1], t[2], t[3]) :
# 		n == 4  ? (t[1], t[2], t[3], t[4]) :
# 		n == 5  ? (t[1], t[2], t[3], t[4], t[5]) :
# 		n == 6  ? (t[1], t[2], t[3], t[4], t[5], t[6]) :
# 		n == 7  ? (t[1], t[2], t[3], t[4], t[5], t[6], t[7]) :
# 		n == 8  ? (t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]) :
# 		n == 9  ? (t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9]) :
# 		n == 10 ? (t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10]) :
# 		begin
# 			v = MVector{n, eletype(T)}(undef)
# 			i = 1
# 			while i <= n
# 				v[i] = t[i]
# 				i += 1
# 			end
# 			Tuple(v)
# 		end
# 	return s
# end



"""
``oneto(n::Integer)``
Construct the tuple `(1,2,...,n)``.
"""
oneto(n::T) where {T<:Integer} = ntuple(identity, n, T)
# When N can be inferred, the following are compile-time generated.
oneto(::Val{0}) = ()
oneto(::Val{N}) where {N} = (oneto(Val(N-1))..., N)
# For Comparison, Base.OneTo produces an iterator, which is usually slower



"""
```tupseq(a::Integer, b::Integer)```
Construct the tuple `(a, a+1, ..., b-1, b)`.
"""
function tupseq(a::Ta, b::Tb) where {Ta<:Integer, Tb<:Integer}
	d = b - a
	t = d < 0  ? () :
		d == 0	? (a,) :
		d == 1  ? (a, a+1) :
		d == 2  ? (a, a+1, a+2) :
		d == 3  ? (a, a+1, a+2, a+3) :
		d == 4  ? (a, a+1, a+2, a+3, a+4) :
		d == 5  ? (a, a+1, a+2, a+3, a+4, a+5) :
		d == 6  ? (a, a+1, a+2, a+3, a+4, a+5, a+6) :
		d == 7  ? (a, a+1, a+2, a+3, a+4, a+5, a+6, a+7) :
		d == 8  ? (a, a+1, a+2, a+3, a+4, a+5, a+6, a+7, a+8) :
		d == 9  ? (a, a+1, a+2, a+3, a+4, a+5, a+6, a+7, a+8, a+9) :
		begin
			v = MVector{d+1, promote_type(Ta,Tb)}(undef)
			x = a
			i = 1
			while i <= d+1
				v[i] = x
				i += 1
				x += 1
			end
			# This is a little slower
			#for (i,x) in pairs(a:b)
			#	v[i] = x
			#end
			# This is the slowest .... not sure how that's possible
			# for i in 1:d+1
			# 	v[i] = x
			# 	x += 1
			# end
			Tuple(v)
		end
	return t
end



"""
```tupseqd(a::Integer, b::Integer)```
Construct the tuple `(a, a-1, ..., b+1, b)`.
"""
function tupseqd(a::Ta, b::Tb) where {Ta<:Integer, Tb<:Integer}
	d = a - b
	t = d < 0  ? () :
		d == 0	? (a,) :
		d == 1  ? (a, a-1) :
		d == 2  ? (a, a-1, a-2) :
		d == 3  ? (a, a-1, a-2, a-3) :
		d == 4  ? (a, a-1, a-2, a-3, a-4) :
		d == 5  ? (a, a-1, a-2, a-3, a-4, a-5) :
		d == 6  ? (a, a-1, a-2, a-3, a-4, a-5, a-6) :
		d == 7  ? (a, a-1, a-2, a-3, a-4, a-5, a-6, a-7) :
		d == 8  ? (a, a-1, a-2, a-3, a-4, a-5, a-6, a-7, a-8) :
		d == 9  ? (a, a-1, a-2, a-3, a-4, a-5, a-6, a-7, a-8, a-9) :
		begin
			v = MVector{d+1, promote_type(Ta,Tb)}(undef)
			x = a
			i = 1
			while i <= d+1
				v[i] = x
				i += 1
				x -= 1
			end
			Tuple(v)
		end
	return t
end


# t[oneto(n)] is about as fast as ntuple(i->t[i], n)


# This is fast due to compile-time evaluation.
"""
`filltup(v, n::Integer)` produces a length-'n' tuple filled with the value 'v'.
"""
filltup(v, n::Integer) = ntuple(i->v, n, typeof(v))
# When this can be inferred it is compile-evaluated.  Otherwise it is SLOW
filltup(v, ::Val{n}) where {n} = Base.fill_to_length((), v, Val(n))


# Concatenate tuples
tcat(t::Tuple) = t
tcat(t::Tuple, tups::Tuple...) = (t..., tcat(tups...)...)

# Indexing tuples with tuples
getindex(t::Tuple, ind::Tuple{}) = ()
getindex(t::Tuple, ind::Tuple{Integer}) = (t[ind[1]],)
getindex(t::Tuple, ind::Tuple{Integer,Integer}) = (t[ind[1]], t[ind[2]])
getindex(t::Tuple, ind::Tuple{Integer,Integer,Integer}) = (t[ind[1]], t[ind[2]], t[ind[3]])
# Defining this method kills type inference...
getindex(t::Tuple, ind::Tuple{Vararg{Integer}}) = map(i->t[i], ind)
# This should be more inferrable than the one above, but it seems to behave exactly the same
# function getindex(t::Tuple, ind::Tuple{Vararg{Integer,N}}) where {N}
# 	a = MVector{N,eltype(t)}(undef)
# 	for i = 1:N
# 		@inbounds a[i] = t[ind[i]]
# 	end
# 	#(a...,)
# 	Tuple(a)
# end
#
#getindex(t::Tuple, ind::Tuple{Vararg{Integer,N}}) where {N} = ntuple(i->t[ind[i]], N)
#getindex(t::Tuple, ind::Tuple{Vararg{Integer}}) = (t[ind[1]], t[tail(ind)]...)

#to_index(I::Tuple{Vararg{Integer}}) = I

# Logical indexing of tuples
# We have to have to separate methods to take precedence over specific methods in Base.
# getindex(t::Tuple, b::AbstractArray{Bool,1}) = getindex_t(t, b)
# getindex(t::Tuple, b::Tuple{Vararg{Bool}}) = getindex_t(t, b)
getindex(t::Tuple, b::Tuple{Vararg{Bool}}) = length(b) == length(t) ? getindex(t, findall(b)) : throw(BoundsError(t, b))

# # As of Julia 1.4, this is no faster than Base's implementation, which uses findall.
# # The type of r is inferred from t
# # Using MVector is a little faster, but loses type flexibility.
# @inline function getindex_t(t::Tuple, b)
# 	if length(b) == length(t)
# 		r = t   # value doesn't matter, just need something the same size as t
# 		#r = MVector{length(t), eltype(t)}(undef)
# 		ir = 0
# 		for i = 1:length(t)
# 			if b[i]
# 				ir += 1
# 				#r[ir] = t[i]
# 				r = Base.setindex(r, t[i], ir)
# 			end
# 			#i += 1
# 		end
# 		return r[oneto(ir)]
# 		#return Tuple(r)[oneto(ir)]
# 		#return Tuple(r[1:ir])
# 	else
# 		throw(BoundsError(t, b))
# 	end
# end


"""
`invperm(p::Tuple)` returns the inverse of permutation `p`.  `p` is not checked.
"""
function invperm(t::Tuple{Vararg{<:Number}})
	p = MVector{length(t), Int}(undef)
	for i = 1:length(t)
		p[t[i]] = i
	end
	Tuple(p)
	#p = t
	#for i = 1:length(t)
	#	p = setindex(p, i, t[i])
	#end
	#p
end


"""
`invpermute(t::Tuple, p::Tuple)` returns `s` such that `s[p] = t`.  `p` is not checked.
"""
function invpermute(t::NTuple{N,Any}, p::NTuple{N,Number}) where {N}
	s = MVector{length(t), Int}(undef)
	for i = 1:length(t)
		s[p[i]] = t[i]
	end
	Tuple(s)
end


# There are 3 ways to partition a tuple t:
#  1. a = t[mask]; b = t[(!).(mask)]
#  2. a = select(t, mask); b = antiselect(t, mask)
#  3. (a, b) = partition_tuple(t, mask)  # (using _partition_tuple)
#  4. (r1, i1, r2, i2) = partition_tuple(t, mask)  # (using __partition_tuple)
#		a = r[oneto(i1)];
#		b = r[oneto(i2)];
#
#  Methods 2 and 4 are the fastest (130 ns); then method 1 (200 ns), then method 3 (400 ns).


# # Actually, this is the fastest!
# """
# `select(t::Tuple, b::AbstractArray{Bool})`
# Selects the elements of `t` at which `b` is `true`. (Equivalent to `t[mask]`.)
# """
# select(t, mask) = t[mask]
#
# """
# `antiselect(t::Tuple, b::AbstractArray{Bool})`
# Selects the elements of `t` at which `b` is `false`. (Equivalent to
# `t[(!).(mask)]` but faster.)
# """
# function antiselect(t, b::Union{AbstractArray{Bool}, Tuple{Vararg{Bool}}})
# 	if length(b) == length(t)
# 		r = t   # value doesn't matter, just need something the same size as t
# 		ir = 0
# 		# Do the logical indexing "by hand".  This is faster than indexing with !b
# 		# because it doesn't allocate a temporary array.
# 		for i = 1:length(t)
# 			if !b[i]
# 				ir += 1
# 				#r[ir] = t[i]
# 				r = Base.setindex(r, t[i], ir)
# 			end
# 		end
# 		return r[oneto(ir)]
# 		#return Tuple(r)[oneto(ir)]
# 	else
# 		throw(BoundsError(t, b))
# 	end
# end
#
#
# """
# `antiselect(t::Tuple, I::AbstractArray{Integer})` Selects the elements of `t` whose
# indices are not in `I`.
#
# This version of `antiselect` internally constructs a logical array.  If such an array is
# already available, using that will be faster.
# """
# function antiselect(t, I::Union{AbstractArray{Integer}, Tuple{Vararg{Integer}}})
# 	b = MVector{length(t),Bool}(undef)
# 	for i in 1:length(t)
# 		b[i] = true
# 	end
# 	for i in I
# 		b[i] = false
# 	end
# 	return t[b]
# end



# """
# Tuple set difference (Not type stable).
# """
# tupdiff(t::NTuple, s::NTuple) = tuple(Iterators.filter(i-> !in(i, s), t)...)
#
# tupintersect(t::NTuple, s::NTuple) = tuple(Iterators.filter(i-> in(i, s), t)...)
#


# This always returns a Bool. (missing counts as false)
function in(v, t::NTuple)
	i = 1
	while i <= length(t)
		if v == t[i]
			return true
		end
		i += 1
	end
	false
end


function findzero(t::Tuple)
	r = t   # value doesn't matter, just need something the same size as t
	ir = 0
	for i = 1:length(t)
		if b[i] == 0
			ir += 1
			#r[ir] = t[i]
			r = Base.setindex(r, t[i], ir)
		end
	end
	return r[oneto(ir)]
	#return Tuple(r)[oneto(ir)]

end

# TODO:  Use sorting when t and/or s are not small
"""
`findin(v, t::NTuple)` returns the index of `v` in `t`.
If `t` contains `v` more than once, the first index is returned.
If `t` does not contain `v`, 0 is returned.

`findin(s::NTuple, t::NTuple)` returns, for each element of `s`, the corresponding
index in `t`.
"""
findin(v, t::NTuple) = find(v .== t)
# For some reason this is faster than map
findin(s::NTuple, t::NTuple) = findin.(s, Ref(t))  # map(v->find(v .== t), s)



"""
Type-stable variant of `findfirst`.

`find(b)` returns the index of the first true element of `b`, or `firstindex(b)-1` if `b`
has no true elements.  This method can be significantly faster than `findfirst` when
the result is placed into a tuple or array.
"""
function find(b)
	i = firstindex(b)
	while i <= lastindex(b)
		if b[i]
			return i
		end
		i += 1
	end
	return firstindex(b)-1
end



function mysort(t::NTuple)
	s = MVector(t)
	sort!(s; alg = QuickSort)
	Tuple(s)
end


function sortperm(t::NTuple)
	s = MVector(oneto(length(t)))
	sortperm!(s, SVector(t); alg = QuickSort)
	Tuple(s)
end



allunique(t::NTuple) = length(t)<8 ? _allunique_by_pairs(t) : _allunique_by_sorting(t)


# Explicitly compare all pairs.  Best choice for small tuples
function _allunique_by_pairs(t::NTuple)
	for i in 1:length(t)
		for j in i+1:length(t)
			if t[j] == t[i]
				return false
			end
		end
	end
	true
end

# TODO:  Write a custom version of quicksort which includes the test for equality.
# That would be faster in the case there is a duplicate element (it bails before sorting
# the whole tuples
# Sort and then compare.  Best choice for not-small tuples.
function _allunique_by_sorting(t::NTuple)
	s = MVector(t)
	sort!(s; alg = QuickSort)
	for i = 2:length(s)
		if s[i] == s[i-1]
			return false
		end
	end
	true
end



# # Tuple arithmetic
# # These appear to be super fast
# (+)(t::Tuple{}, x::Number) = ()
# (-)(t::Tuple{}, x::Number) = ()
# (*)(t::Tuple{}, x::Number) = ()
#
# (+)(t::Tuple{Vararg{T,N}}, x::Number) where {T,N} = ntuple(i->t[i]+x, N)
# (-)(t::Tuple{Vararg{T,N}}, x::Number) where {T,N} = ntuple(i->t[i]-x, N)
# (*)(t::Tuple{Vararg{T,N}}, x::Number) where {T,N} = ntuple(i->t[i]*x, N)
#
# (+)(a::Tuple{Vararg{TA,N}}, b::Tuple{Vararg{TB,N}}) where {TA,TB,N} = ntuple(i->a[i]+b[i], N)
# (-)(a::Tuple{Vararg{TA,N}}, b::Tuple{Vararg{TB,N}}) where {TA,TB,N} = ntuple(i->a[i]-b[i], N)
# (*)(a::Tuple{Vararg{TA,N}}, b::Tuple{Vararg{TB,N}}) where {TA,TB,N} = ntuple(i->a[i]*b[i], N)


#splittup(t::Tuple, n::Integer) = (t[oneto(n)], t[tupseq(n+1,length(t))])

# These all become slow when not inferred
# splittup(::Val{0}, t...) = ((), t)
# splittup(::Val{1}, a, t...) = ((a,), t)
# splittup(::Val{2}, a1, a2, t...) = ((a1,a2), t)
# splittup(::Val{3}, a1, a2, a3, t...) = ((a1,a2,a3), t)
# splittup(::Val{4}, a1, a2, a3, a4, t...) = ((a1,a2,a3,a4), t)
# splittup(::Val{5}, a1, a2, a3, a4, a5, t...) = ((a1,a2,a3,a4,a5), t)
# splittup(::Val{6}, a1, a2, a3, a4, a5, a6, t...) = ((a1,a2,a3,a4,a5,a6), t)
# splittup(::Val{7}, a1, a2, a3, a4, a5, a6, a7, t...) = ((a1,a2,a3,a4,a5,a6,a7), t)
# splittup(::Val{8}, a1, a2, a3, a4, a5, a6, a7, a8, t...) = ((a1,a2,a3,a4,a5,a6,a7,a8), t)
# splittup(::Val{9}, a1, a2, a3, a4, a5, a6, a7, a8, a9, t...) = ((a1,a2,a3,a4,a5,a6,a7,a8,a9), t)
# splittup(::Val{10}, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, t...) = ((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10), t)
#
# split_(::Val{N}, a) where {N} = split_(Val(N-1), (a[1],), Base.tail(a))
# split_(::Val{N}, a, b) where {N} = split_(Val(N-1), (a..., b[1]), Base.tail(b))
# split_(::Val{0}, a, b) = (a, b)

# """
# Partition a tuple into two tuples according to a logical mask.
# `partition_tuple(t, mask) = (a,b)` where `a = a[findall(mask)]` and `b = b[findall(!mask)]`.
# """
# partition_tuple(t::Tuple, b::AbstractArray{Bool}) = _partition_tuple(t, b)
# partition_tuple(t::Tuple, b::Tuple{Vararg{Bool}}) = _partition_tuple(t, b)
#
# function _partition_tuple(t::Tuple, b)
# 	if length(b) == length(t)
# 		# r1 = MVector{length(t), eltype(t)}(undef)
# 		# r2 = MVector{length(t), eltype(t)}(undef)
# 		r1 = t
# 		r2 = t
# 		i1 = 0
# 		i2 = 0
# 		for it = 1:length(t)
# 			if b[it]
# 				i1 += 1
# 				#r1[i1] = t[it]
# 				r1 = Base.setindex(r1, t[it], i1)
# 			else
# 				i2 += 1
# 				#r2[i2] = t[it]
# 				r2 = Base.setindex(r2, t[it], i2)
# 			end
# 		end
# 		return (r1, i1, r2, i2)						# this is the fastest
# 		#return (r1[oneto(i1)], r2[oneto(i2)])		# this is the slowest
# 	else
# 		throw(BoundsError(t, b))
# 	end
# end
#
# # This function returns a tuple the same size as t, but with the "true" values
# # up front and the "false" values at the back, and with the index of the number
# # of "true" values.  This can be used to reconstruct the partition.  It is
# # slightly less convenient that the function above, but is the fastest of the
# function __partition_tuple(t::Tuple, b)
# 	if length(b) == length(t)
# 		r = t
# 		i1 = 0
# 		i2 = length(t)+1
# 		for it = 1:length(t)
# 			if b[it]
# 				i1 += 1
# 				#r1[i1] = t[it]
# 				r = Base.setindex(r, t[it], i1)
# 			else
# 				i2 -= 1
# 				#r2[i2] = t[it]
# 				r = Base.setindex(r, t[it], i2)
# 			end
# 		end
# 		return (r, i1)
# 		#return (Tuple(r1)[oneto(i1)], Tuple(r2)[oneto(i2)])
# 	else
# 		throw(BoundsError(t, b))
# 	end
# end



# # This is slow, because length(b) is not inferrable.
# # If we pass a compile-time constant for the length, it becomes as fast as getindex_t.
# function findall(b::AbstractArray{Bool,1})
# 	r = MVector{length(b), Int64}(undef)
# 	ir = 0
# 	for i = 1:length(b)
# 		if b[i]
# 			ir += 1
# 			r[ir] = i
# 		end
# 	end
# 	return Tuple(r)[oneto(ir)]
# end

# findin1(t::Dims, v::Int64) = findin1(t, (v,))
# function findin1(t::Dims, s::Tuple{Vararg{Int64,N}}) where {N}
# 	#tmap = Vector{Int64}(undef, maximum(t))
# 	tmap = zeros(Int64, maximum(t))
# 	#tmap = MVector{maximum(t),Int64}(undef) #slow!
# 	#map(i- tmap[i] = i, t)
# 	for i in 1:length(t)
# 		tmap[t[i]] = i
# 	end
# 	return map(v->tmap[v], s)
# end

#findin2(t::Dims, v::Int64) = firstequal(v, t)


#findin2(x::Tuple{Vararg{Int64}}, s::Tuple{Vararg{Int64}}) = filter(i->notnothing(i), map(v->findfirst(v .== s), x))
#


# function my_ntuple_rec(f, n::Int64, i::Int64)
#     if i<n
#         return (f(i), my_ntuple_rec(f, n, i+1)...)
#     else
#         return (f(n),)
#     end
# end
#
# my_ntuple(f, vn::Type{Val{n}}) where {n} = my_ntuple_rec(f, vn, Val{1})
# my_ntuple_rec(f, vn::Type{Val{n}}, vi::Type{Val{i}}) where {i,n} = i<n ? (f(i), my_ntuple_rec(f, vn, Val{i+1})...) : f(n)




end
