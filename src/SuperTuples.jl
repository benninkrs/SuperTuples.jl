# Additions/extensions to Julia that I (RSB) like
module SuperTuples

export oneto, tupseq, tupseqd, filltup, tupseqiter, accumtuple, invpermute
export findin, tfindall, tcat #, indexed_union
#export ntuple_, map_

using StaticArrays
using Base: tail
import StaticArrays: deleteat
import Base: getindex, setindex
import Base: ntuple, invperm, sort, sortperm, findfirst
import Base: cumsum


# TODO:  accumtuple()
# - Make special method of accumtuple for NTuples of same type as x0
# - Figure out which methods are fasted and dispatch to fastest one
#
# TODO: getindex, select, findall
# - figure out which methods are fastest and dispatch to fastest one


# Equivalent to Base._foldoneto
"""
`static_fn(op, a, Val(n))` statically computes f_n(⋯f_1(a)) where f_i(x) = op(x, i)

[It seems that one could achieve the same thing by just defining op recursively ...?]
"""
@inline function static_fn(op, acc, ::Val{N}) where N
   @assert N::Integer > 0
   if @generated
      quote
         acc_0 = acc
         Base.Cartesian.@nexprs $N i -> acc_{i} = op(acc_{i-1}, i)
         return $(Symbol(:acc_, N))
      end
   else
      for i in 1:N
         acc = op(acc, i)
      end
      return acc
   end
end


# Tuple constructors

# # On 1.6.0 this appears to be obsolete. -- Base ntuple is faster
# """
# 	ntuple(f, n, T)

# Create an `NTuple{N,T}` with values `(f(1), ..., f(n))`. For n>10, this is much faster
# than ntuple(f, n).  An `InexactError` will occur if any of `f(1)`,...,`f(n)` cannot be
# converted to type `T`.
# """
# function ntuple_(f, n::Integer, ::Type{T}) where {T}
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
#          begin
#       v = MVector{n,T}(undef)
#       @inbounds for i in 1:n
#          v[i] = f(i)
#       end
#       Tuple(v)
#    end
# end

# function ntuple_(f, ::Val{n}, ::Type{T}) where {n} where {T}
#    t = n == 0 ? () :
#        n == 1 ? ((f(1),)) :
#        n == 2 ? ((f(1), f(2))) :
#        n == 3 ? ((f(1), f(2), f(3))) :
#        n == 4 ? ((f(1), f(2), f(3), f(4))) :
#        n == 5 ? ((f(1), f(2), f(3), f(4), f(5))) :
#        n == 6 ? ((f(1), f(2), f(3), f(4), f(5), f(6))) :
#        n == 7 ? ((f(1), f(2), f(3), f(4), f(5), f(6), f(7))) :
#        n == 8 ? ((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8))) :
#        n == 9 ? ((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9))) :
#        n == 10 ? ((f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10))) :
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
   fk = k -> static_fn((y, j) -> j <= k ? f(y) : y, x, Val(n))
   ntuple(fk, Val(n))
end


function cumsum(t::Tuple{Vararg{Integer}})
   f = i -> static_fn((x, j) -> j <= i ? x + t[j] : x, 0, Val(length(t)))
   ntuple(f, Val(length(t)))
end

function cumsum(t::Tuple{Vararg{Float64}})
   f = i -> static_fn((x, j) -> j <= i ? x + t[j] : x, 0, Val(length(t)))
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


# For some reason this is slower than using static_fn:
# cumtuple(f, x, ::Val{0}) = ()
# function cumtuple(f, x, ::Val{n}) where {n}
# 	(f(x), _cumtuple(f, f(x), Val(n-1))...)
# end


# This is faster than cumtuple:
# ntuple(i->static_fn((x,j)-> j<=i ? f(x) : x, 1, Val(25)), Val(25))


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
oneto(n::T) where {T<:Integer} = ntuple(identity, n)
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
filltup(v, n::Integer) = ntuple(i -> v, n)
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
# getindex(t::Tuple, inds::Tuple{Vararg{Integer}}) = map(i->t[i], inds)

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


# Logical indexing of tuples
# We have to have to separate methods to take precedence over specific methods in Base.
getindex(t::Tuple, b::Tuple{Vararg{Bool}}) = getindex_(t, b)
const ManyBool = Tuple{Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Vararg{Bool}}
getindex(t::Tuple{Vararg{T}}, b::ManyBool) where {T} = getindex_(t, b)


function getindex_(t::Tuple, b)
   # length(b) == length(t) ? getindex(t, tfindall(b)) : throw(BoundsError(t, b))
   length(b) == length(t) || throw(BoundsError(t, b))
   # replicating the code from tfindall here makes it faster
   n = length(t)
   c = cumsum(b)
   m = c[end]
   ntuple(i->t[findin(c, i)], m)
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
         static_fn(t[i], Val(M)) do a, j
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



# On Julia 1.6, map is slow for length(inds) >= 16.
# How can we overload it?
# map_(f, t::Tuple{Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]))
# map_(f, t::Tuple{Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]))
# map_(f, t::Tuple{Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]))
# map_(f, t::Tuple{Any,Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]))
# map_(f, t::Tuple{Any,Any,Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8]))
# map_(f, t::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8]), f(t[9]))
# map_(f, t::Tuple{Any,Any,Any,Any,Any,Any,Any,Any,Any,Any}) = (f(t[1]), f(t[2]), f(t[3]), f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8]), f(t[9]), f(t[10]))



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
      static_fn(x0, Val(M)) do a, j
         # this bounds check slows things down a lot
         # (idx[j] > N || idx[j] < 0) && error("Index $(idx[j]) is invalid for a tuple of length $N")
         idx[j] == i ? op(a, v[j]) : a
      end
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



# # Closer to Base's version.  Slow when x0 is of different type than t
# function accumtup_(v::NTuple{M}, idx::Dims{M}, x0, ::Val{N}, op) where {M,N}
#    ntuple(Val(N)) do i
#       a = static_fn(nothing, Val(N)) do a, j
#          a !== nothing && return a
#          idx[j] == i && return op(a, v[j])
#          nothing
#       end
#       a === nothing ? x0 : a
#    end
# end




# Base has an implementation of invperm(::Tuple) which falls back to invperm(::Vector) for n>=16.
# This version uses the Base code for N<=20, and a significantly faster method for n>=20.
function invperm(p::Tuple{Vararg{<:Integer,N}}, b) where {N}
   if N <= 20
      #return accumtuple(oneto(Val(N)), p, 0, Val(N), (x,y) -> x==0 ? y : 0)
      ntuple(Val(N)) do i
         s = static_fn(nothing, Val(N)) do s, j
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




"""
	invpermute(t::Tuple, p::Tuple)

Returns `s` such that `s[p] == t`, or equivalently `s = t[invperm(p)]`.
If `p` is longer than 32, its validity as a permutation is not checked.

See also [`findin`](@ref) and [`accumtuple`](@ref).
"""
function invpermute(t::NTuple{N,Any}, p::NTuple{N,<:Integer}) where {N}
   if N <= 32
     #      return accumtuple_short(t, p, nothing, Val(N), replace_nothing)
      ntuple(Val(N)) do i
         a = static_fn(nothing, Val(N)) do a, j
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



# Compiler-inferrable version for tuples
function findfirst(::Val{b}) where {b}
   N = length(b)
   static_fn(nothing, Val(N)) do i, j
      i !== nothing && return i
      b[j] && return j
      nothing
   end
end



"""
`findin(t::Tuple, v)` returns the index of `v` in `t`.
If `t` contains `v` more than once, the first index is returned.
If `t` does not contain `v`, an error is thrown.

`i = findin(t::Tuple, s::Tuple)` returns, for each element of `s`, the corresponding
index in `t`, so that `t[i] = s``.
"""
function findin(t::NTuple{N,<:Integer}, v) where {N}
   # Just do a straightforward search.
   # Using sort to find elements is not advantageous for any reasonably-sized tuple
   i = static_fn(nothing, Val(N)) do i, i_
      i !== nothing && return i
      t[i_] == v && return i_
      nothing
   end
   i === nothing && throw(ArgumentError("v was not found in t"))
   i
end

# in principle we could broadcast the scalar version ... ?
function findin(t::NTuple{N,<:Integer}, s::NTuple{M,Any}) where {N,M}
   ntuple(Val(M)) do j
      i = static_fn(nothing, Val(N)) do a, i_
         a !== nothing && return a
         t[i_] == s[j] && return i_
         nothing
      end
      i === nothing && throw(ArgumentError("s[$j] was not found in t"))
      i
   end
end


# this is the fastest of various methods.  Fast for m<=10, slower for m>10.
# n = m = 10:  ~25ns
# n = m = 11:  ~250ns
# For t large and only partially true, findall is sometimes faster
function tfindall(t::Tuple{Vararg{Bool}})
   n = length(t)
   c = cumsum(t)
   m = c[end]
   #ntuple(i->findin(c, i), m)
   #ntuple(i->findin(c, i), Val(m))
   #ntuple_(i->findin(c, i), m, Int)
   #(findall(t)...,)
   #([i for i in oneto(m)]...,)
   if m<=10
      ntuple(i->findin(c, i), m)
   else
      tfindall_(t, c, Val(m))
      #ntuple(i->findin(c, i), Val(m))
      #tfindall(Val(t))
      #(findall(t)...,)
   end
end

tfindall_(t::Tuple{Vararg{Bool}}, c, ::Val{m}) where m = ntuple(i->findin(c, i), Val(m))

# This works just as well as the complicated version below
@generated function tfindall(::Val{t}) where {t}
   i = (findall(t)...,)
   return :( $i )
end


# function tfindall(::Val{t}) where {t}  # how to restrict t to Tuple{Vararg{Bool}}?
#    t isa Tuple{Vararg{Bool}} || error("t must be a tuple of Bools")
#    n = length(t)

#    # version 1 - 0ns
#    c = cumsum(t)
#    m = c[end]
#    ntuple(i->findin(i, c), m)

#    # version 2: - 1.6us
#    # m = sum(t)
#    # v = MVector{m,Int}(undef)
#    # k = 1
#    # @inbounds for i in 1:n
#    #    if t[i]
#    #       v[k] = i
#    #       k += 1
#    #    end
#    # end
#    # Tuple(v)

#    # version 3: - 40ns
#    # m = numtrue(t)
#    # v = Vector{Int}(undef, m)
#    # k = 1
#    # @inbounds for i in 1:n
#    #    if t[i]
#    #       v[k] = i
#    #       k += 1
#    #    end
#    # end
#    # ntuple(i->v[i], Val(m))

#    # version 4 - 17us
#    # m = sum(t)
#    # v = oneto(m)
#    # k = 1
#    # @inbounds for i in 1:n
#    #    if t[i]
#    #       v = setindex(v, i, k)
#    #       k += 1
#    #    end
#    # end
#    # v
# end





#-------------------------------------
# Sorting



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


#-------------------------------
# Set operations

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


end
