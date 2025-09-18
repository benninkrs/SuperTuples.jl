# Additions/extensions to Julia that I (RSB) like
module SuperTuples

export oneto, tuplerange, setindex, accumtuple, cumfun
export findin, tfindall, tcat #, indexed_union
export ntuple_iter_f_x, ntuple_iter_f_ix, ntuple_iter_seq
# export deleteat

using StaticArrays
using Base: tail
# import StaticArrays: deleteat
import Base: getindex, setindex
import Base: ntuple, invperm, sort, sortperm
import Base: cumsum, cumprod
export tail

@warn "Package SuperTuples extends some functions in Base to enable new uses of tuples 
      and to increase the performance of some functions on tuples. 
      Theoretically this could result in unexpected behavior of Base or other packages."



# Faster version when the output type is known.
"""
   ntuple(f, n, ::Type)
Construct a tuple with known type. For n≤10 this methods performs the
same as the untyped method in Base; but for n>10 it is significantly faster.
An error results if the output of f cannot be converted to `type`.

Note: the type is not used or enforced for n≤10.
"""
function ntuple(f, n, ::Type{T}) where {T}
   t = n == 0 ? () :
       n == 1 ? (f(1),) :
       n == 2 ? (f(1), f(2)) :
       n == 3 ? (f(1), f(2), f(3)) :
       n == 4 ? (f(1), f(2), f(3), f(4)) :
       n == 5 ? (f(1), f(2), f(3), f(4), f(5)) :
       n == 6 ? (f(1), f(2), f(3), f(4), f(5), f(6)) :
       n == 7 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7)) :
       n == 8 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)) :
       n == 9 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9)) :
       n == 10 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10)) :
   begin
      v = MVector{n,T}(undef)
      @inbounds for i in 1:n
         v[i] = f(i)
      end
      Tuple(v)
   end
end


"""
   ntuple(f, Val(n), type)
Construct a tuple of known, uniform type.  For n≤10 this methods performs the
same as the untyped method in Base; but for n>10 it is significantly faster.
An error results if the output of f cannot be converted to `type`.
"""
function ntuple(f, ::Val{n}, ::Type{T}) where {n,T}
   t = n == 0 ? () :
       n == 1 ? (f(1),) :
       n == 2 ? (f(1), f(2)) :
       n == 3 ? (f(1), f(2), f(3)) :
       n == 4 ? (f(1), f(2), f(3), f(4)) :
       n == 5 ? (f(1), f(2), f(3), f(4), f(5)) :
       n == 6 ? (f(1), f(2), f(3), f(4), f(5), f(6)) :
       n == 7 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7)) :
       n == 8 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)) :
       n == 9 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9)) :
       n == 10 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10)) :
   begin
      v = MVector{n,T}(undef)
      @inbounds for i in 1:n
         v[i] = f(i)
      end
      Tuple(v)
   end
end


# function ntuple_(f, n)
#    t = n == 0 ? () :
#        n == 1 ? (f(1),) :
#        n == 2 ? (f(1), f(2)) :
#        n == 3 ? (f(1), f(2), f(3)) :
#        n == 4 ? (f(1), f(2), f(3), f(4)) :
#        n == 5 ? (f(1), f(2), f(3), f(4), f(5)) :
#        n == 6 ? (f(1), f(2), f(3), f(4), f(5), f(6)) :
#        n == 7 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7)) :
#        n == 8 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8)) :
#        n == 9 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9)) :
#        n == 10 ? (f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9), f(10)) :
#    begin
#       T = eltype(f(1))
#       v = MVector{n,T}(undef)
#       # this loop is very slow
#       @inbounds for i in 1:n
#          f_i = f(i)
#          T_ = typeof(f_i)
#          if T_ <: T
#             v[i] = f_i
#          else
#             v_ = MVector{n,promote_type(T,T_)}(undef)
#             for j = 1:i-1
#                v_[j] = v[j]
#             end
#             v_[i] = f_i
#             v = v_
#          end
#       end
#       Tuple(v)
#    end
# end


"""
	oneto(n::Integer)
	oneto(::Val{n})

Construct the tuple `(1,2,...,n)`.  When `n` is inferrable the constructions above
usually leads to faster code than `1:n` or `Base.OneTo(n)`.  But when `n` is not inferrable,
the latter constructions are more performant.
"""
oneto(n::T) where {T<:Integer} = ntuple(identity, n, T)

# When N can be inferred, the following are compile-time generated.
oneto(::Val{0}) = ()
oneto(::Val{N}) where {N} = (oneto(Val(N - 1))..., N)

# t[oneto(n)] is about as fast as ntuple(i->t[i], n)


"""
   tuplerange(a::Integer, b::Integer)

Construct tuple((a:b)...) directly, without constructing the intermediate range.
"""
function tuplerange(a::Integer, b::Integer)
   T = promote_type(typeof(a), typeof(b))
   len = max(0, b-a+oneunit(T))
   ntuple(i -> (a-oneunit(T)) + i, len, T)
end




# # This is fast due to compile-time evaluation.
# """
#    filltup(v, n::Integer)
#    filltup(v, ::Val(n))
# produces a length-'n' tuple filled with the value 'v'.
# """
# filltup(v, n::Integer) = ntuple(i -> v, n)
# # When this can be inferred it is compile-evaluated.  Otherwise it is SLOW
# filltup(v, ::Val{n}) where {n} = Base.fill_to_length((), v, Val(n))



# Internal Utilities

#=
Here's a sneaky way to create compiler-inferrable tuples from iterative functions.

Say we have a function f and want to create a tuple t such that t[i] = f(t[i-1]) with t[1] = f(t0).
Naivley we could achieve this by the following:

function iter_n(f, a, i)
   for j = 1:i
      a = f(a)
   end
   a
end

t = ntuple(i -> iter_n(f, t0, i))

But the result of iter_n is not compiler-inferrable, since i is not a compiler constant.
Hence it will not compile to conise code.

[EDIT:  In Julia 1.10, it appears that we don't need the "sneaky" approach below.  We can
define iter_n to take a Val(i) argument and do
   t = ntuple(i -> iter_n(f, t0, Val(i)))
and this performs as well as the "sneaky" method below.
]

The sneaky approach used in base is to evalute a fixed number of iterations of a wrapper function
that switches from f to the identity when the desired number of iterations has been reached:
function iter_n(f, a, i, ::Val{n}) where {n}
   for j = 1:n
      j <= i ? a = f(a) : a
   end
   a
end

t = ntuple(i -> iter_n(f, t0, i, Val(n)))


=#



"""
   static_iter_f_x(f, x0, Val(n))

Computes, in a compiler-inferrable way, the result of iterating a function f(x) n times.
Equivalent to:
   x1 = f(x0)
   x2 = f(x1)
   ...
   xn = f(x(n-1))
   return xn

This is mainly used by [`ntuple_iter_x`](@ref) to construct tuples from stateful functions. 
"""
@inline function static_iter_f_x(f, a, ::Val{N}) where {N}
   @assert N::Integer > 0
   if @generated
      quote
         a_0 = a
         Base.Cartesian.@nexprs $N j -> (a_{j} = f(a_{j-1}))
         return $(Symbol(:a_, N))
      end
   else
      for j = 1:N
         a = f(a)
      end
      return a
   end
end
static_iter_f_x(f, a, ::Val{0}) = a



"""
   ntuple_iter_f_x(f::Function, x0, Val(n))

Create a tuple `t` from `n` iterations of a function `f(x)`:
   t[1] = f(x0)
   t[2] = f(t[1])
   ...
   t[n] = f(t[n-1])
See also [`ntuple_iter_ix`](@ref) and [`ntuple_iter_s`](@ref)
"""
function ntuple_iter_f_x(f::F, x0, ::Val{n}) where {F<:Function, n}
   # One could also build the tuple recursively, i.e. (f(x0), ntuple_iter(f, f(x0), n-1)...)
   # That has the same performance as the "static iter" approach for n<=33,
   # but is MUCH worse for n>33.
   ntuple(i -> static_iter_f_x(f, x0, Val(i)), Val(n))
end


# Surprisingly, this ispretty performant (Julia 1.10) but using static_iter is faster for larger n.
# function ntuple_iter_x_2(f::F, x0, ::Val{n}) where {F<:Function, n}
#    iter_f = k -> begin
#          x = x0
#          for i = 1:k
#             x = f(x)
#          end
#          return x
#       end 
#    ntuple(i -> iter_f(i), Val(n))
# end


# function ntuple_iter_f_x(f::F, x0, n) where {F<:Function}
#    iter_f = k -> begin
#          x = x0
#          for i = 1:k
#             x = f(x)
#          end
#          return x
#       end 
#    ntuple(i -> iter_f(i), n)
# end






"""
   static_iter_f_ix(f, x0, Val(n))
Computes, in a compiler-inferrable way, the result of iterating a function f(i,x) n times.
Equivalent to:
   x1 = f(1, x0)
   x2 = f(2, x1)
   ...
   xn = f(n, x(n-1))
   return xn

This is mainly used by [`ntuple_iter_ix`](@ref) to construct tuples from stateful functions. 
"""
@inline function static_iter_f_ix(f, a, ::Val{N}) where {N}
   @assert N::Integer > 0
   if @generated
      quote
         a_0 = a
         Base.Cartesian.@nexprs $N j -> (a_{j} = f(j, a_{j-1}))
         return $(Symbol(:a_, N))
      end
   else
      for j = 1:N
         a = f(j, a)
      end
      return a
   end
end
static_iter_f_ix(f, a, ::Val{0}) = a


"""
   ntuple_iter_f_ix(iterfun::Function, x0, n)
   ntuple_iter_f_ix(iterfun::Function, x0, Val(n))

Create a tuple `t` from `n` iterations of a function `f`:
   t[1] = f(1, x0)
   t[2] = f(2, t[1])
   ...
   t[n] = f(n, t[n-1])
See also [`ntuple_iter_ix`](@ref) and [`ntuple_iter_s`](@ref)
"""
function ntuple_iter_f_ix(f::F, x0, ::Val{n}) where {F<:Function, n}
   ntuple(i -> static_iter_f_ix(f, x0, Val(i)), Val(n))
end





"""
   static_iter_seq(seq, n)
   static_iter_seq(seq, Val(n))

Compute, in a compiler-inferrable way, the `n`th value of iteratable `seq`.
Equivalent to:
   (x1, s1) = iterate(seq)
   (x2, s2) = iterate(seq, s2)
   ...
   (xn, sn) = iterate(seq, s(n-1))
   return xn

This is mainly used by [`ntuple_iter_seq`](@ref) to construct tuples from non-indexable iterators. 
"""
@inline function static_iter_seq(seq, ::Val{N}) where {N} 
   @assert N::Integer > 0
   # Should we do sequence length check here?
   # The code below will error if N > length(seq)
   # Instead we would have to do:
   #  r_1 = iterate(seq)
  #    isnothing(r_1) || ((x_1, s_1) = r_1)
  #    ...
   if @generated
      quote
         (x_1, s_1) = iterate(seq)
         Base.Cartesian.@nexprs $(N-1) j -> (x_{j+1}, s_{j+1}) = iterate(seq, s_{j})   
         return $(Symbol(:x_, N))
      end
   else
      (x,s) = iterate(seq)
      for j = 2:N
         (x,s) = iterate(seq, s)
      end
      return x
   end
end




"""
   ntuple_iter_seq(seq::Iterable)
   ntuple_iter_seq(seq::Iterable, Val(n))

Create a tuple `t` such that `t[i]` is the `i`th value of iterable `seq`.
This is primarily useful when `seq` is not indexable.

The second form specifies the output length `n`, which may be shorter or longer than `seq`.
(For i > length(seq), t[i] = seq[end].)  This form can be much faster than the first when `n`
is inferrable.
"""
function ntuple_iter_seq(seq, ::Val{n}) where {n}
   ntuple(i -> static_iter_seq(seq, Val(i)), Val(n))
end
ntuple_iter_seq(seq) = ntuple_iter_seq(seq, Val(length(seq)))




# Replaces Base's version.
# This is much faster.
function cumsum(t::Tuple)
   ntuple_iter_f_ix((i,s) -> s+t[i], zero(eltype(t)), Val(length(t)))
end

function cumprod(t::Tuple)
   ntuple_iter_f_ix((i,s) -> s*t[i], one(eltype(t)), Val(length(t)))
end

"""
   cumfun(t, op)

Like cumsum or cumprod, but accumulating values using binary operation `op`.
"""
function cumfun(t::Tuple, op)
   if length(t) <= 1
      t
   else
      (t[1], ntuple_iter_f_ix((i,s) -> op(s, t[i+1]), t[1], Val(length(t)-1))...)
   end
   #ntuple_iter((i,s) -> (i==1) ? t[i] : op(s, t[i]), nothing, Val(length(t)))
end

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

const ManyIntegers = Tuple{Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer, Vararg{Integer}}
const ManyBool = Tuple{Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Bool, Vararg{Bool}}

# A faster version for large uniformly-typed input
function getindex(t::Tuple{Vararg{T}}, inds::ManyIntegers) where {T}
   all(1 <= i <= length(t) for i in inds) || throw(BoundsError(t, inds))
   N = length(inds)
   r = MVector{N,T}(undef)
   for i in 1:N
      @inbounds r[i] = t[inds[i]]
   end
   Tuple(r)
end


# Logical indexing of tuples
# We have to have to separate methods to prevent ambiguity
@inline getindex(t::Tuple, b::Tuple{Vararg{Bool}}) = getindex(t, tfindall(b))
@inline getindex(t::Tuple, b::ManyBool) = getindex(t, tfindall(b))
@inline getindex(t::Tuple{Vararg{T}}, ::Tuple{}) where {T} = ()
@inline getindex(t::Tuple{Vararg{T}}, b::Tuple{Vararg{Bool}}) where {T} = getindex(t, tfindall(b))
# @inline getindex(t::Tuple{Vararg{T}}, b::ManyBool) where {T} = getindex(t, tfindall(b))
@inline getindex(t::Tuple{Vararg{T}}, b::ManyBool) where {T} = getindex_(t, b)

# [v1.10] Somehow const prop works better this way
function getindex_(t::Tuple{Vararg{T}}, b::Tuple{Vararg{Bool}}) where {T}
   # (copied from tfindall)
   length(b) == length(t) || throw(BoundsError(t, b))
   n = length(t)
   c = cumsum(b)
   m = c[end]
   ntuple(i -> t[findin(c, i)], m, T)
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
   @boundscheck all(1 <= i <= N for i in idx) || throw(BoundsError(t, idx))
   if N <= 32
      return ntuple(Val(N)) do i
         @inbounds static_iter_f_ix(t[i], Val(M)) do j, a
            idx[j] == i ? v[j] : a
         end
      end
   else
      s = MVector(t)
      for i = 1:length(idx)
         @inbounds s[idx[i]] = v[i]
      end
      return Tuple(s)
   end
end



# !! Should probably require v and x0 to be compatible types.  (When they are not, performance is slow ... I think type inference fails)
"""
accumtuple(vals::Tuple, inds, x0, n::Integer, acc = +)

Construct a tuple of length `n` by accumulating values `v` at indices `i`.
`x0` is the value assigned to elements not indexed by `i`.
`acc` is the accumulation function:  t[i] ⟵ acc(t[i], v).

See also  ['setindex'](@ref).
"""
accumtuple(v, idx, x0, n, op=+) = accumtuple(v, idx, x0, Val(n), op)

function accumtuple(v, idx, x0, ::Val{N}, op=+) where {N}
   all(1 <= i <= N for i in idx) || throw(BoundsError)
   N <= 20 ? accumtuple_short(v, idx, x0, Val(N), op) :
   accumtuple_long(v, idx, x0, Val(N), op)
end



# This is kind of slow, compared to, say, Base.invperm.  Why?
function accumtuple_short(v::NTuple{M}, idx::Dims{M}, x0, ::Val{N}, op) where {M,N}
   @inbounds ntuple(Val(N)) do i
      static_iter_f_ix(x0, Val(M)) do j, a
         idx[j] == i ? op(a, v[j]) : a
      end
   end
end


function accumtuple_long(v::NTuple{M}, idx::Dims{M}, x0, ::Val{N}, op) where {M,N}
   p = fill(x0, (N, 1))    # faster than an MVector in this case.
   @inbounds for i in 1:M
      j = idx[i]
      p[j] = op(p[j], v[i])
   end
   return ntuple(i -> p[i], Val(N))
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





# Base has an implementation of invperm(::Tuple) which falls back to invperm(::Vector) for n>=16.
# This version is essentailly the same as the Base code for N<=20, and a significantly faster method for n>=20.
function invperm(p::Tuple{Vararg{<:Integer,N}}) where {N}
   if N <= 20
      #return accumtuple(oneto(Val(N)), p, 0, Val(N), (x,y) -> x==0 ? y : 0)
      ntuple(Val(N)) do i
         s = static_iter_f_ix(nothing, Val(N)) do j, s
            s !== nothing && return s
            @inbounds p[j] == i && return j
            nothing
         end
         s === nothing && throw(ArgumentError("argument is not a permutation"))
         s
      end
   else
      all(1 <= p[i] <= N for i in p) || throw(BoundsError)
      invp = MVector{length(p),Int}(undef)
      @inbounds for i = 1:length(p)
         invp[p[i]] = i
      end
      return Tuple(invp)
   end
end



# # Do we need this?  Why not just use t[invperm(p)]?
# """
# 	invpermute(t::Tuple, p::Tuple)

# Returns `s` such that `s[p] == t`, or equivalently `s = t[invperm(p)]`.
# If `p` is longer than 32, its validity as a permutation is not checked.

# See also [`findin`](@ref) and [`accumtuple`](@ref).
# """
# function invpermute(t::NTuple{N,Any}, p::NTuple{N,<:Integer}) where {N}
#    if N <= 32
#      #      return accumtuple_short(t, p, nothing, Val(N), replace_nothing)
#       ntuple(Val(N)) do i
#          a = static_iter(nothing, Val(N)) do j, a
#             a !== nothing && return a
#             @inbounds p[j] == i && return t[j]
#             nothing
#          end
#          a === nothing && throw(ArgumentError("p is not a permutation"))
#          a
#       end
#    else
#       s = SizedVector{length(t)}(Vector{Any}(undef, N))
#       for i = 1:length(t)
#          @inbounds s[p[i]] = t[i]
#       end
#       return Tuple(s)
#    end
# end




# """
# 	deleteat(t::Tuple, I::Integer)
# 	deleteat(t::Tuple, I::Iterable{Integer})

# Selects the elements of `t` whose indices are not in `I`.
# (Logical indexing is generally faster if the logical array already exists).
# """
# function deleteat(t::Tuple, i::Integer)
#    1 <= i <= length(t) || throw(BoundsError)
#    length(t) <= 33 ? _deleteat(t, i) : _deleteat_long(t, i)
# end


# function deleteat(t::Tuple, I::Tuple{Integer,Integer,Vararg{Integer}})
#    any(i -> !(1 <= i <= length(t)), I) && throw(BoundsError)
#    length(t) <= 33 ? _deleteat(t, sort(I, rev=true)) : _deleteat_long(t, sort(I, rev=true))
# end

# @inline _deleteat(t::Tuple, i::Int) = i == 1 ? tail(t) : (t[1], _deleteat(tail(t), i - 1)...)

# @inline _deleteat(t::Tuple, I::Tuple{Integer}) = _deleteat(t, I[1])
# @inline _deleteat(t::Tuple, I::Tuple{Integer,Integer,Vararg{Integer}}) =
#    _deleteat(_deleteat(t, I[1]), tail(I)) # assumes sorted from big to small


# _deleteat_long(t::Tuple, I::Integer) = _deleteat_long(t, (I,))
# function _deleteat_long(t::Tuple, I::Union{AbstractArray{Integer},Tuple{Vararg{Integer}}})
#    b = MVector{length(t),Bool}(undef)
#    for i in 1:length(t)
#       b[i] = true
#    end
#    for i in I
#       b[i] = false
#    end
#    return t[b]
# end


# Base already has a fast implementation.  No need to implement our own.
# function findfirst(b::Tuple{Vararg{Bool}})
#    N = length(b)
#    return static_iter_f_ix(nothing, Val(N)) do i,j
#       # i indexes b.  j is the matched index (or nothing)
#       (isnothing(j) && b[i]) ? i : j 
#    end
# end




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

   # The obtuse code below implements the following:
   #
   # j = nothing
   # for i = 1:N
   #  t[i] == v && j = i
   # end
   #
   # ... but does it in such a way that the compiler unrolls the loop,
   # which results in much faster (?) run-time performance.

   j = static_iter_f_ix(nothing, Val(N)) do i,j
      # i indexes t.  j is the matched index (or nothing)
      (isnothing(j) && t[i] == v) ? i : j 
   end

	# If we allow nothing to be returned, the result is type unstable
	# and functions like tfindall become much slower.
   j === nothing && throw(ArgumentError("v was not found in t"))
   j
end

# in principle we could broadcast the scalar version ... ?
function findin(t::NTuple{N,<:Integer}, v::NTuple{M,Any}) where {N,M}
   ntuple(Val(M)) do k
      j = static_iter_f_ix(nothing, Val(N)) do i,j
         # i indexes t.  j is the matched index (or nothing)
         (isnothing(j) && t[i] == v[k]) ? i : j 
      end

		# If we allow nothing to be returned, the result is type unstable
		# and functions like tfindall become much slower.
      j === nothing && throw(ArgumentError("v was not found in t"))
      j
      end
end


# Fast for m<=10, much slower for m>10.
# If the result will have more than 10 elements
# Non-const propagated:
#	n = m = 10:  ~25 ns
# 	n = m = 11:  115 ns
# 	n = m = 20:  4.6 μs
# Const-propagatd:
#	n = m = 10:  ~8 ns
# 	n = m = 11:  90 ns
#	n = m = 20:	4.6 μs
# TODO:  Is it possible to do better?
# - Dispatching on Val(sum(t)) is slower.
# - Recursive tuple construction is even slower.
# Probably not possible to do better, since oneto(n) with n=11 takes essentially
# the same time.  That is, the slowdown is not findin() but simply the fact
# that the length (and hence type) of the result is not inferrable.
"""
   tfindall(t::Tuple{Vararg{Bool}})

Return a tuple containing the indices, in order, of all `true` elements of `t`.
(In contrast `findall` returns a `Vector`.) `tfindall` is much faster when
the number of returned elements is <=10; otherwise `findall` is much faster.
"""
function tfindall(t::Tuple{Vararg{Bool}})
   n = length(t)
   c = cumsum(t)
   m = c[end]        # total number of true elements
   if m<=10
      ntuple(i->findin(c, i), m)
   else
      ntuple(i->findin(c, i), Val(m))
   end
end




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


# #-------------------------------
# # Set operations

# allunique(t::NTuple) = length(t) <= 7 ? _allunique_by_pairs(t) : _allunique_by_sorting(t)

# # Explicitly compare all pairs.  Best choice for small tuples
# function _allunique_by_pairs(t::NTuple)
#    for i in 1:length(t)
#       for j in i+1:length(t)
#          if t[j] == t[i]
#             return false
#          end
#       end
#    end
#    true
# end

# # TODO:  Write a custom version of quicksort which includes the test for equality.
# # That would be faster in the case there is a duplicate element (it bails before sorting
# # the whole tuples
# # Sort and then compare.  Best choice for not-small tuples.
# function _allunique_by_sorting(t::NTuple)
#    s = MVector(t)
#    sort!(s; alg=QuickSort)
#    for i = 2:length(s)
#       if s[i] == s[i-1]
#          return false
#       end
#    end
#    true
# end


# """
# 	(u, i1, i2) = index_union(t1::Tuple, t2::Tuple)

# Returns a vector `u` that is the sorted union of elements in `t1`,`t2` and tuples
# `i1`,`i2` such that `t1 = u[i1]` and `t2 = u[i2]`.
# """
# function indexed_union(t1::NTuple, t2::NTuple)
#    if length(t1) == 0
#       return (t1, (), oneto(length(t2)))
#    elseif length(t2) == 0
#       return (t2, oneto(length(t1)), ())
#    else
#       t12 = tcat(t1, t2)
#       U = promote_type(eltype(t1), eltype(t2))
#       N = length(t1) + length(t2)

#       # sort all the elements
#       perm = sortperm(t12)
#       iperm = invpermute(oneto(N), perm)
#       s12 = t12[perm]


#       # Extract unique elements and their indices in t1, t2
#       u = MVector{N,U}(undef)
#       #u = s12	# value doesn't matter, just need a tuple of same size as s12
#       s12_to_u = MVector{N,U}(undef)

#       u[1] = s12[1]
#       #u = setindex(u, s12[1], 1)
#       s12_to_u[1] = 1
#       j = 1
#       for i in 2:N
#          if s12[i] > s12[i-1]
#             j += 1
#             u[j] = s12[i]
#             #setindex(u, s12[i], j)
#          end
#          s12_to_u[i] = j
#       end

#       i1 = ntuple(i -> s12_to_u[iperm[i]], Val(length(t1)))
#       i2 = ntuple(i -> s12_to_u[iperm[i+length(t1)]], Val(length(t2)))
#       # For u as a MVector:
#       return (u[1:j], i1, i2)# 301 ns
#       #		return (Tuple(u)[1:j], i1, i2)	# 764 ns
#       #		return (Tuple(u)[oneto(j)], i1, i2)	# 890 ns
#       #		return (Tuple(u[1:j]), i1, i2)	# 840 ns
#       #return
#    end
# end





# Obsolete stuff


# @inline function static_iterated_fn(op, state_0, ::Val{N}) where N
#    @assert N::Integer > 0
#    if @generated
#       expr = quote
#          Base.Cartesian.@nexprs $N i -> (i==1) ? (value_{i}, state_{i}) = op(zero(typeof(state_0)), state_{i-1}, i) : (value_{i}, state_{i}) = op(value_{i-1}, state_{i-1}, i)
#          return $(Symbol(:value_, N))
#       end
#       else
#          error("not generated")
#    #    (value, state) = op(nothing, state_0, 1)
#    #    for i in 2:N
#    #       (value, state) = op(value, state, i)
#    #    end
#    #    return value
#    end
# end


# @inline function static_iterated_fn(op, state_0, k, ::Val{N}) where N
#    @assert N::Integer > 0
#    @assert k::Integer > 0
#    # if @generated
#    #    expr = quote
#    #       Base.Cartesian.@nexprs $N i -> (value_{i}, state_{i}) = (i<=k) ? op(state_{i-1}, i) : (value_{k}, state_{k})
#    #       return $(Symbol(:value_, N))
#    #    end
#    #   println("generated")
#    #   show(expr)
#    #    expr
#    # else
#       #println("not generated")
#       (value, state) = op(state_0, 1)
#       for i in 2:N
#          (value, state) = (i<=k) ? op(state, i) : (value, state)
#       end
#       return value
#    # end
# end


#
# An attempt to define a convenient way to generate a tuple from a stateful iterator.
# Sadly, it doesn't infer at all.  But maybe it can be rewritten using static_fn.
#

# struct TupleIterator{F <: Function, N, T}
#    iterfun::F
#    seed::T
#    TupleIterator(f::F, s::T, ::Val{N}) where {F,T,N} = new{F,N,T}(f, s)
# end

# Base.length(::TupleIterator{F,N}) where {F,N} = N
# function Base.iterate(ti::TupleIterator{F}) where {F}
#    (x, s) = ti.iterfun(1, ti.seed)
#    (x, (2, s))
# end
# function Base.iterate(ti::TupleIterator{F,N}, (i, s)) where {F,N}
#    if (i <= N)
#       (x, s) = ti.iterfun(i, s)
#       return (x, (i+1, s))
#    else
#       return nothing
#    end
# end

# # Saldy, this doesn't infer
# function ntuple_iter(f::F, s0, ::Val{N}) where {F<:Function, N}
#    ti = TupleIterator(f, s0, Val(N))
#    tuple(ti...)
# end


# function cumsum_(t::Tuple{Vararg{T}}) where {T}
#    ntuple_iter((i,s) -> (i==1) ? (t[i], t[i]) : (s+t[i], s+t[i]), zero(T), Val(length(t)))
# #   ntuple_iter((s,i) -> (s+t[i], s+t[i]), zero(T), Val(length(t)))
# end

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

# old way
# function tupseqiter(f, x, ::Val{n}) where {n}
#    # fk applies f k times
#    fk = k -> static_fn((y, j) -> j <= k ? f(y) : y, x, Val(n))
#    ntuple(fk, Val(n))
# end


# old way
# function cumsum(t::Tuple{Vararg{T}}) where {T}
#    f = i -> static_fn((x, j) -> j <= i ? x + t[j] : x, zero(T), Val(length(t)))
#    ntuple(f, Val(length(t)))
# end


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


end
