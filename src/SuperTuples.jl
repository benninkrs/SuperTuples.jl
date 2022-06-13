# Additions/extensions to Julia that I (RSB) like
module SuperTuples

export oneto, tupseq, tupseqd, filltup, tupseqiter, accumtuple, invpermute
export firsttrue, findin, tcat, indexed_union

using StaticArrays
using Base: tail
import StaticArrays: deleteat
import Base: getindex, setindex, map
import Base: ntuple, invperm, sort, sortperm
import Base: cumsum


# TODO:  accumtuple()
# - Make special method of accumtuple for NTuples of same type as x0
# - Figure out which methods are fasted and dispatch to fastest one
#
# TODO: getindex, select, findall
# - figure out which methods are fastest and dispatch to fastest one


# Tuple constructors

"""
	ntuple(f, n, T)

Create an `NTuple{N,T}` with values `(f(1), ..., f(n))`. For n>10, this is much faster
than ntuple(f, n).  An `InexactError` will occur if any of `f(1)`,...,`f(n)` cannot be
converted to type `T`.
"""
function ntuple(f, n::Integer, ::Type{T}) where {T}
   t = n == 0 ? () :
       n == 1 ? NTuple{n,T}((f(1),)) :
       n == 2 ? NTuple{n,T}((f(1), f(2))) :
       n == 3 ? NTuple{n,T}((f(1), f(2), f(3))) :
       n == 4 ? NTuple{n,T}((f(1), f(2), f(3), f(4))) :
       n == 5 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5))) :
       n == 6 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6))) :
       n == 7 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7))) :
       n == 8 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8))) :
       n == 9 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9))) :
       n == 10 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10))) :
   begin
      v = MVector{n,T}(undef)
      @inbounds for i in 1:n
         v[i] = f(i)
      end
      Tuple(v)
   end
end

# function ntuple(f, ::Val{n}, ::Type{T}) where {n} where {T}
#    t = n == 0 ? () :
#        n == 1 ? NTuple{n,T}((f(1),)) :
#        n == 2 ? NTuple{n,T}((f(1), f(2))) :
#        n == 3 ? NTuple{n,T}((f(1), f(2), f(3))) :
#        n == 4 ? NTuple{n,T}((f(1), f(2), f(3), f(4))) :
#        n == 5 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5))) :
#        n == 6 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6))) :
#        n == 7 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7))) :
#        n == 8 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8))) :
#        n == 9 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9))) :
#        n == 10 ? NTuple{n,T}((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10))) :
#    begin
#       v = MVector{n,T}(undef)
#       @inbounds for i in 1:n
#          v[i] = f(i)
#       end
#       Tuple(v)
#    end
# end

# function ntuple_(f, n::Integer)
# #	n == 0 ? () : (f(n), ntuple_(f, n-1)...)
# 	ntuple_(f, Val(n))
# end
#
# function ntuple_(f, ::Val{n}) where {n}
# 	n == 0 ? () : (ntuple_(f, Val(n-1))..., f(n))
# end
#

"""
	tupseqiter(f, x0, n)
	tupseqiter(f, x0, Val{n})

Construct the tuple `(f(x0), f(f(x0)), ..., f^n(x0))`.
"""
tupseqiter(f, x, n::Integer) = tupseqiter(f, x, Val{n})
function tupseqiter(f, x, ::Val{n}) where {n}
   # fk applies f k times
   fk = k -> Base._foldoneto((y, j) -> j <= k ? f(y) : y, x, Val(n))
   ntuple(fk, Val(n))
end


function cumsum(t::Tuple)
   f = i -> Base._foldoneto((x, j) -> j <= i ? x + t[j] : x, false, Val(length(t)))
   ntuple(f, Val(length(t)))
end


# SLOW!
# cumsumtup_(t::Tuple{}) = ()
# cumsumtup_(t::Tuple) = (t[1], cumsumtup_(Base.tail(t), t[1])...) 
# cumsumtup_(t::Tuple{}, v) = ()
# function cumsumtup_(t::Tuple, v)
# 	v += t[1]
# 	return (v, cumsumtup_(Base.tail(t), v)...)
# end


# For some reason this is slower than using Base._foldoneto:
# cumtuple(f, x, ::Val{0}) = ()
# function cumtuple(f, x, ::Val{n}) where {n}
# 	(f(x), _cumtuple(f, f(x), Val(n-1))...)
# end


# This is faster than cumtuple:
# ntuple(i->Base._foldoneto((x,j)-> j<=i ? f(x) : x, 1, Val(25)), Val(25))


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
	oneto(n::Integer)
	oneto(::Val{n})

Construct the tuple `(1,2,...,n)`.  When `n` is inferrable the constructions above
usually leads to faster code than `1:n` or `Base.OneTo(n)`.  But when `n` is not inferrable, the latter constructions are preferred.
"""
oneto(n::T) where {T<:Integer} = ntuple(identity, n, T)
# When N can be inferred, the following are compile-time generated.
oneto(::Val{0}) = ()
oneto(::Val{N}) where {N} = (oneto(Val(N - 1))..., N)
# For Comparison, Base.OneTo produces an iterator, which is usually slower

oneto(::Type{Val{0}}) = ()
oneto(::Type{Val{N}}) where {N} = (oneto(Val{N - 1})..., N)


"""
```tupseq(a::Integer, b::Integer)```
Construct the tuple `(a, a+1, ..., b-1, b)`.
"""
function tupseq(a::Ta, b::Tb) where {Ta<:Integer,Tb<:Integer}
   d = b - a
   t = d < 0 ? () :
       d == 0 ? (a,) :
       d == 1 ? (a, a + 1) :
       d == 2 ? (a, a + 1, a + 2) :
       d == 3 ? (a, a + 1, a + 2, a + 3) :
       d == 4 ? (a, a + 1, a + 2, a + 3, a + 4) :
       d == 5 ? (a, a + 1, a + 2, a + 3, a + 4, a + 5) :
       d == 6 ? (a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6) :
       d == 7 ? (a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7) :
       d == 8 ? (a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7, a + 8) :
       d == 9 ? (a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7, a + 8, a + 9) :
   begin
      v = MVector{d + 1,promote_type(Ta, Tb)}(undef)
      x = a
      i = 1
      while i <= d + 1
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
function tupseqd(a::Ta, b::Tb) where {Ta<:Integer,Tb<:Integer}
   d = a - b
   t = d < 0 ? () :
       d == 0 ? (a,) :
       d == 1 ? (a, a - 1) :
       d == 2 ? (a, a - 1, a - 2) :
       d == 3 ? (a, a - 1, a - 2, a - 3) :
       d == 4 ? (a, a - 1, a - 2, a - 3, a - 4) :
       d == 5 ? (a, a - 1, a - 2, a - 3, a - 4, a - 5) :
       d == 6 ? (a, a - 1, a - 2, a - 3, a - 4, a - 5, a - 6) :
       d == 7 ? (a, a - 1, a - 2, a - 3, a - 4, a - 5, a - 6, a - 7) :
       d == 8 ? (a, a - 1, a - 2, a - 3, a - 4, a - 5, a - 6, a - 7, a - 8) :
       d == 9 ? (a, a - 1, a - 2, a - 3, a - 4, a - 5, a - 6, a - 7, a - 8, a - 9) :
   begin
      v = MVector{d + 1,promote_type(Ta, Tb)}(undef)
      x = a
      i = 1
      while i <= d + 1
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
filltup(v, n::Integer) = ntuple(i -> v, n, typeof(v))
# When this can be inferred it is compile-evaluated.  Otherwise it is SLOW
filltup(v, ::Type{Val{n}}) where {n} = Base.fill_to_length((), v, Val(n))


# Concatenate tuples
tcat(t::Tuple) = t
tcat(t::Tuple, tups::Tuple...) = (t..., tcat(tups...)...)

# Indexing tuples with tuples
getindex(t::Tuple, i::Tuple{}) = ()
getindex(t::Tuple, i::Tuple{Integer}) = (t[i[1]],)
getindex(t::Tuple, i::Tuple{Integer,Integer}) = (t[i[1]], t[i[2]])
getindex(t::Tuple, i::Tuple{Integer,Integer,Integer}) = (t[i[1]], t[i[2]], t[i[3]])
getindex(t::Tuple, i::Tuple{Integer,Integer,Integer,Integer}) = (t[i[1]], t[i[2]], t[i[3]], t[i[4]])
getindex(t::Tuple, inds::Tuple{Vararg{Integer}}) = ntuple(i -> t[inds[i]], Val(length(inds)))

# A faster version for large uniformly-typed tuples
const ManyIntegers = Tuple{Integer,Integer,Integer,Integer,Integer,
   Integer,Integer,Integer,Integer,Vararg{Integer}}
function getindex(t::Tuple{Vararg{T}}, inds::ManyIntegers) where {T}
   N = length(inds)
   r = MVector{N,T}(undef)
   for i in 1:N
      r[i] = @inbounds t[inds[i]]
   end
   Tuple(r)
end


# Base has setindex(tuple, value, index), but not setindex(tuple, values, tuple)
"""
	setindex(t::Tuple, v, i::Dims)

Construct a tuple equivalent to `t[i] = v`.
See also  ['accumtuple'](@ref) and [`invpermute`](@ref).
"""
@inline function setindex(t::Tuple, v, idx::Dims)
   N = length(t)
   M = length(idx)
   if N <= 32
      return ntuple(Val(N)) do i
         Base._foldoneto(t[i], Val(M)) do a, j
            @boundscheck 1 <= idx[j] <= N || throw(BoundsError(t, idx[j]))
            idx[j] == i ? v[j] : a
         end
      end
   else
      s = MVector(t)
      for i = 1:length(idx)
         s[idx[i]] = v[i]
      end
      return Tuple(s)
   end
end

# Logical indexing of tuples
# We have to have to separate methods to take precedence over specific methods in Base.
# getindex(t::Tuple, b::AbstractArray{Bool,1}) = getindex_t(t, b)
# getindex(t::Tuple, b::Tuple{Vararg{Bool}}) = getindex_t(t, b)
getindex(t::Tuple, b::Tuple{Vararg{Bool}}) = length(b) == length(t) ? getindex(t, findall(b)) : throw(BoundsError(t, b))
getindex(t::Tuple{Vararg{T}}, b::Tuple{Vararg{Bool}}) where {T} = length(b) == length(t) ? getindex(t, findall(b)) : throw(BoundsError(t, b))
const ManyBool = Tuple{Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Vararg{Bool}}
getindex(t::Tuple{Vararg{T}}, b::ManyBool) where {T} = length(b) == length(t) ? getindex(t, findall(b)) : throw(BoundsError(t, b))


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


# Type inferred
#findall_(b::Tuple{Vararg{Bool}}) = findall_(Val(b))

# # Very slow!
# function findall_(b::Tuple{Vararg{Bool}})
# 	n = sum(b)
# 	cumb = cumsum(b) .* b
# 	ntuple(Val(n)) do i
# 		# t[i]
# 		# find the index of the ith true element
# 		Base._foldoneto(0, Val(length(b))) do j, k
# 			cumb[k] == i ? k : j
# 		end
# 	end
# end

# # Also slower than findall
# function findall_(::Val{b}) where {b}
# 	isempty(b) || b isa Tuple{Vararg{Bool}} || error("b must be a tuple of Bools")
# 	n = sum(b)
# 	cumb = cumsum(b) .* b
# 	ntuple(Val(n)) do i
# 		# t[i]
# 		# find the index of the ith true element
# 		Base._foldoneto(0, Val(length(b))) do j, k
# 			cumb[k] == i ? k : j
# 		end
# 	end
# end


# select is a tentative replacement for getindex(tuple, bool_tuple)

# Not type inferred
select(t::Tuple, mask::Tuple{Vararg{Bool}}) = select(t, Val(mask))

function select(t::Tuple, ::Val{mask}) where {mask}
	length(mask) == length(t) || error("t and mask must have the same length")
	n = sum(mask)
	cummask = cumsum(mask) .* mask
	ntuple(Val(n)) do i
		# t[i]
		# find the index of the ith true element
		j = Base._foldoneto(0, Val(length(t))) do j_, k
			cummask[k] == i ? k : j_
		end
		t[j]
	end
end


# select_(t::Tuple, ::Val{()}) = ()
# function select_(t::Tuple, ::Val{mask}) where {mask}
# 	if mask[1]
# 		return (t[1], select_(Base.tail(t), Val(Base.tail(mask)))...)
# 	else
# 		return select_(Base.tail(t), Val(Base.tail(mask)))
# 	end
# end




# On Julia 1.5, map is slow for length(inds) >= 16
# getindex(t::Tuple, inds::Tuple{Vararg{Integer}}) = map(i->t[i], inds)
map(f, t::Tuple{Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]))
map(f, t::Tuple{Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]))
map(f, t::Tuple{Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]))
map(f, t::Tuple{Any,Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]))
map(f, t::Tuple{Any,Any,Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8]))



# !! Should probably require v and x0 to be compatible types.  (When they are not, performance is slow ... I think type inference fails)
"""
	accumtuple(v::Tuple, i::Dims, x0, n, accumfun = +)

Construct a tuple of length `n` by accumulating values `v` at indices `i`.
`x0` is the value assigned to elements not indexed by `i`.
`accumfun(x, x_new)` is a binary function used to accumulate values.

See also  ['setindex'](@ref) and [`invpermute`](@ref).
"""
accumtuple(v, idx, x0, n, op=+) = accumtuple(v, idx, x0, Val(n), op)

function accumtuple(v, idx, x0, ::Val{N}, op=+) where {N}
   N <= 32 ? accumtuple_short(v, idx, x0, Val(N), op) :
   accumtuple_long(v, idx, x0, Val(N), op)
end



# This is kind of slow, compared to, say, Base.invperm.  Why?
function accumtuple_short(v::NTuple{M}, idx::Dims{M}, x0, ::Val{N}, op) where {M,N}
   ntuple(Val(N)) do i
      Base._foldoneto(x0, Val(M)) do a, j
         # this bounds check slows things down a lot
         # (idx[j] > N || idx[j] < 0) && error("Index $(idx[j]) is invalid for a tuple of length $N")
         idx[j] == i ? op(a, v[j]) : a
      end
   end
end


# Closer to Base's version.  Slow when x0 is of different type than t
function accumtup_(v::NTuple{M}, idx::Dims{M}, x0, ::Val{N}, op) where {M,N}
   ntuple(Val(N)) do i
      a = Base._foldoneto(nothing, Val(N)) do a, j
         a !== nothing && return a
         idx[j] == i && return op(a, v[j])
         nothing
      end
      a === nothing ? x0 : a
   end
end

function accumtuple_long(v::NTuple{M}, idx::Dims{M}, x0, ::Val{N}, op) where {M,N}
   p = fill(x0, (N, 1))
   for i in 1:M
      j = idx[i]
      p[j] = op(p[j], v[i])
   end
   return ntuple(i -> p[i], Val(N))
end




# Base has an implementation of invperm(::Tuple) which falls back to invperm(::Vector) for n>=16.
# This version uses the Base code for N<=20, and a significantly faster method for n>=20.
function invperm(p::Tuple{Vararg{<:Integer,N}}, b) where {N}
   if N <= 20
      #return accumtuple(oneto(Val(N)), p, 0, Val(N), (x,y) -> x==0 ? y : 0)
      ntuple(Val(N)) do i
         s = Base._foldoneto(nothing, Val(N)) do s, j
            s !== nothing && return s
            p[j] == i && return j
            nothing
         end
         s === nothing && throw(ArgumentError("argument is not a permutation"))
         s
      end
   else
      invp = MVector{length(p),Int}(undef)
      for i = 1:length(p)
         invp[p[i]] = i
      end
      return Tuple(invp)
   end
end

# Replace an element only if it has not been set already
#replace_nothing(::Nothing, x) = x
#replace_nothing(x, y) = error("Indices are not all unique")



"""
	invpermute(t::Tuple, p::Tuple)

Returns `s` such that `s[p] = t`.  If `p` is longer than 32, its validity as a permutation is not checked.

See also [`accumtuple`](@ref).
"""
function invpermute(t::NTuple{N,Any}, p::NTuple{N,<:Integer}) where {N}
   if N <= 32
     #      return accumtuple_short(t, p, nothing, Val(N), replace_nothing)
      ntuple(Val(N)) do i
         a = Base._foldoneto(nothing, Val(N)) do a, j
            a !== nothing && return a
            p[j] == i && return t[j]
            nothing
         end
         a === nothing && throw(ArgumentError("p is not a permutation"))
         a
      end
   else
      s = SizedVector{length(t)}(Vector{Any}(undef, N))
      for i = 1:length(t)
         s[p[i]] = t[i]
      end
      return Tuple(s)
   end
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
# function deleteat(t, b::Union{AbstractArray{Bool}, Tuple{Vararg{Bool}}})
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
"""
	deleteat(t::Tuple, I::Integer)
	deleteat(t::Tuple, I::Iterable{Integer})

Selects the elements of `t` whose indices are not in `I`.
(Logical indexing is generally faster if the logical array already exists).
"""
deleteat(t::Tuple, I::Tuple{Int}) = deleteat(t, I[1])
function deleteat(t::Tuple, i::Int)
   1 <= i <= length(t) || throw(BoundsError(t, i))
   length(t) <= 33 ? _deleteat(t, i) : _deleteat_long(t, i)
end


function deleteat(t::Tuple, I::Tuple{Integer,Integer,Vararg{Integer}})
   any(i -> !(1 <= i <= length(t)), I) && throw(BoundsError(t, I))
   length(t) <= 33 ? _deleteat(t, sort(I, rev=true)) : _deleteat_long(t, sort(I, rev=true))
end

@inline _deleteat(t::Tuple, i::Int) = i == 1 ? tail(t) : (t[1], _deleteat(tail(t), i - 1)...)

@inline _deleteat(t::Tuple, I::Tuple{Integer}) = _deleteat(t, I[1])
@inline _deleteat(t::Tuple, I::Tuple{Integer,Integer,Vararg{Integer}}) =
   _deleteat(_deleteat(t, I[1]), tail(I)) # assumes sorted from big to small


_deleteat_long(t::Tuple, I::Integer) = _deleteat_long(t, (I,))
function _deleteat_long(t::Tuple, I::Union{AbstractArray{Integer},Tuple{Vararg{Integer}}})
   b = MVector{length(t),Bool}(undef)
   for i in 1:length(t)
      b[i] = true
   end
   for i in I
      b[i] = false
   end
   return t[b]
end

# """
# Tuple set difference (Not type stable).
# """
# tupdiff(t::NTuple, s::NTuple) = tuple(Iterators.filter(i-> !in(i, s), t)...)
#
# tupintersect(t::NTuple, s::NTuple) = tuple(Iterators.filter(i-> in(i, s), t)...)
#


# This always returns a Bool. (missing counts as false)
# function in(v, t::NTuple)
# 	i = 1
# 	while i <= length(t)
# 		if v == t[i]
# 			return true
# 		end
# 		i += 1
# 	end
# 	false
# end


function findzero(t::Tuple)
   r = t   # value doesn't matter, just need something the same size as t
   ir = 0
   for i = 1:length(t)
      if t[i] == 0
         ir += 1
         #r[ir] = t[i]
         r = Base.setindex(r, t[i], ir)
      end
   end
   return r[oneto(ir)]
   #return Tuple(r)[oneto(ir)]

end

# TODO:  Considuer using _foldoneto, as in accumtuple
# Using sort to find elements is not advantageous for any reasonably-sized tuple
"""
`findin(v, t::NTuple)` returns the index of `v` in `t`.
If `t` contains `v` more than once, the first index is returned.
If `t` does not contain `v`, 0 is returned.

`findin(s::NTuple, t::NTuple)` returns, for each element of `s`, the corresponding
index in `t`.
"""
findin(v, t::NTuple) = firsttrue(v .== t)



"""
Type-stable variant of `findfirst`.

`firsttrue(b)` returns the index of the first true element of `b`, or `firstindex(b)-1` if `b`
has no true elements.  This method can be significantly faster than `findfirst` when
the result indexes a tuple or array.
"""
@inline function firsttrue(b)
   i = firstindex(b)
   @inbounds while i <= lastindex(b)
      if b[i]
         return i
      end
      i += 1
   end
   return firstindex(b) - 1
end


# If the tuple is short, use compiler-inferred merge sort.
# Otherwise use quicksort with an MVector scratch space.
# A switchpoint of length(t)==15 would actually be better, but for some reason
# setting higher then 9 causes major performance hits for selected length tuples (even
# though the called functions are individually fast).
sort(t::Tuple; lt=isless, by=identity, rev::Bool=false) = length(t) <= 9 ? _sort(t, lt, by, rev) : _sort_long(t, lt, by, rev)

# Taken from TupleTools.jl
@inline function _sort(t::Tuple, lt=isless, by=identity, rev::Bool=false)
   t1, t2 = _split(t)
   t1s = _sort(t1, lt, by, rev)
   t2s = _sort(t2, lt, by, rev)
   return _merge(t1s, t2s, lt, by, rev)
end
_sort(t::Tuple{Any}, lt=isless, by=identity, rev::Bool=false) = t
_sort(t::Tuple{}, lt=isless, by=identity, rev::Bool=false) = t

function _split(t::NTuple{N}) where {N}
   M = N >> 1
   ntuple(i -> t[i], M), ntuple(i -> t[i+M], N - M)
end

function _merge(t1::Tuple, t2::Tuple, lt, by, rev)
   if lt(by(first(t1)), by(first(t2))) != rev
      return (first(t1), _merge(tail(t1), t2, lt, by, rev)...)
   else
      return (first(t2), _merge(t1, tail(t2), lt, by, rev)...)
   end
end
_merge(::Tuple{}, t2::Tuple, lt, by, rev) = t2
_merge(t1::Tuple, ::Tuple{}, lt, by, rev) = t1
_merge(::Tuple{}, ::Tuple{}, lt, by, rev) = ()


@inline function _sort_long(t::Tuple, lt=isless, by=identity, rev::Bool=false)
   s = MVector{length(t),eltype(t)}(t)
   sort!(s; lt=lt, by=by, rev=rev, alg=QuickSort)
   Tuple(s)
end


sortperm(t::Tuple; lt=isless, by=identity, rev::Bool=false) = length(t) <= 9 ? _sortperm(t, lt, by, rev) : _sortperm_long(t, lt, by, rev)

# Adapted from TupleTools
function _sortperm(t::Tuple, lt=isless, by=identity, rev::Bool=false)
   _sort(oneto(length(t)), lt, i -> by(t[i]), rev)
end

function _sortperm_long(t::Tuple, lt=isless, by=identity, rev::Bool=false)
   s = MVector(oneto(length(t)))
   sort!(s, lt=lt, by=i -> by(t[i]), rev=rev, alg=QuickSort)
   Tuple(s)
end



allunique(t::NTuple) = length(t) <= 7 ? _allunique_by_pairs(t) : _allunique_by_sorting(t)

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
   sort!(s; alg=QuickSort)
   for i = 2:length(s)
      if s[i] == s[i-1]
         return false
      end
   end
   true
end

"""
	(u, i1, i2) = index_union(t1::Tuple, t2::Tuple)

Returns a vector `u` that is the sorted union of elements in `t1`,`t2` and tuples
`i1`,`i2` such that `t1 = u[i1]` and `t2 = u[i2]`.
"""
function indexed_union(t1::NTuple, t2::NTuple)
   if length(t1) == 0
      return (t1, (), oneto(length(t2)))
   elseif length(t2) == 0
      return (t2, oneto(length(t1)), ())
   else
      t12 = tcat(t1, t2)
      U = promote_type(eltype(t1), eltype(t2))
      N = length(t1) + length(t2)

      # sort all the elements
      perm = sortperm(t12)
      iperm = invpermute(oneto(N), perm)
      s12 = t12[perm]


      # Extract unique elements and their indices in t1, t2
      u = MVector{N,U}(undef)
      #u = s12	# value doesn't matter, just need a tuple of same size as s12
      s12_to_u = MVector{N,U}(undef)

      u[1] = s12[1]
      #u = setindex(u, s12[1], 1)
      s12_to_u[1] = 1
      j = 1
      for i in 2:N
         if s12[i] > s12[i-1]
            j += 1
            u[j] = s12[i]
            #setindex(u, s12[i], j)
         end
         s12_to_u[i] = j
      end

      i1 = ntuple(i -> s12_to_u[iperm[i]], Val(length(t1)))
      i2 = ntuple(i -> s12_to_u[iperm[i+length(t1)]], Val(length(t2)))
      # For u as a MVector:
      return (u[1:j], i1, i2)# 301 ns
      #		return (Tuple(u)[1:j], i1, i2)	# 764 ns
      #		return (Tuple(u)[oneto(j)], i1, i2)	# 890 ns
      #		return (Tuple(u[1:j]), i1, i2)	# 840 ns
      #return
   end
end

# function indexed_union(tups::Vararg{NTuple, M}) where M
# 	lengths = map(length, tups)
# 	bigtup = tcat(tups...)
# 	N = length(bigtup)
# 	isort = sortperm(bigtup)
# 	bigsorted = bigtup[isort]
# 	indS = MVector{Int,N}(undef)
# 	ind_bs[1] = 1
# 	for i in 2:N
# 		if ind_bs[i] > ind_bs[i-1]
# 			ind_bs[i] = ind_bs[i-1]+1
# 		else
# 			ind_bs[i] = ind_bs[i]
# 		end
# 	end
#
# 	ind_bs = ind_bs[isort]
# error("unfinished!")
# end

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
