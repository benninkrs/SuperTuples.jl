# SuperTuples.jl
Utilities for working with tuples in Julia

**SuperTuples** provides convenient, performant methods for working with tuples:

* construction from iterated functions and other tuples
* concatenation
* indexing via other tuples
* sorting and permuting
* searching

Some of the new functionality is achieved by adding methods to standard Base functions (such as `getindex`). Since this constitutes [type piracy](https://docs.julialang.org/en/v1/manual/style-guide/#Avoid-type-piracy-1) there is the possibility that **using SuperTuples may lead to unexpected behavior in unrelated code (including Base)**, though this seems unlikely at present.

SuperTuples provides some similar functionality as, but was developed mostly independently of, [Tuple Tools](https://github.com/Jutho/TupleTools.jl).

# Tuple Construction

## Constructing Ranges
```
oneto(n)
oneto(Val(n))
```
Construct the tuple (1,2,...,n).  When `n` is inferrable or these
usually lead to faster code than `1:n` or `Base.OneTo(n)`.


```
tuplerange(a, b)
```
Constructs the tuple `(a, ..., b)` taking steps of size `oneunit(a)`.


```
ntuple_iter(seq)
ntuple_iter(seq, Val(n))
```
Create a tuple `t` such that `t[i]` is the `i`th value of iterable `seq`.
This is primarily useful when `seq` is not indexable.

The second form specifies the output length `n`, which may be shorter or longer than `seq`.
(For i > length(seq), t[i] = seq[end].)  This form can be much faster than the first when `n` is inferrable.

### Examples

```
julia> oneto(5)
(1,2,3,4,5)

julia> tuplerange(2,6)
(2,3,4,5,6)

julia> tuplerange(7.0,3)
(7.0, 6.0, 5.0, 4.0, 3.0)
```


## Constructing from Functions
```
ntuple(f, n::Integer, type)
ntuple(f, Val(n), type)
```
Construct a tuple of known, uniform type from callable object `f`.
For n≤10 these methods perform the same as the untyped methods in Base;
but for n>10 they are significantly faster.
An error results if the output of f cannot be converted to `type`.


```
ntuple_iter_f_x(f, x0, Val(n))
```
Construct a tuple by iterating function `f(x)` on the initial value `x0`. The resulting tuple
`t` has `t[i] = f(t[i-1])` where `t[0] ≡ x0`.  


```
ntuple_iter_f_ix(f, x0, Val(n))
```
Construct a tuple by iterating function `f(i,x)` on the initial value `x0`. The resulting
tuple `t` has `t[i] = f(i, t[i-1])` where `t[0] ≡ x0`.  



```
ntuple_iter_seq(seq)
ntuple_iter_seq(seq, Val(n))
```
Create a tuple from iterable object `seq`. The result is a tuple `t` such that `t[i]` is the `i`th value of `seq`. This is primarily useful when `seq` is not indexable.

The second form specifies the output length `n`, which may be shorter or longer than `seq`.
(For i > length(seq), `t[i] = seq[end]`.)  This form can be much faster than the first when `n` is inferrable.





## Construction from Other Tuples
SuperTuples provides several functions for constructing tuples by selectively placing elements from other tuples.  As described below, `getindex` can be used to extract tuple elements and `setindex` can be used effectively modify selected elements of a tuple.

A generalization of such functionality is provided by `accumtuple`: 
```
accumtuple(vals::Tuple, inds, x0, n::Integer, acc = +)
accumtuple(vals::Tuple, inds, x0, Val(n), acc = +)
```
constructs a tuple of length `n` by accumulating values `vals` at indices `inds`.
`x0` is the value assigned to elements not indexed by `i`.
`acc(x, y)` is a binary function used to accumulate values, i.e. for each `i` in `inds`, `t[i] = acc(t[i], v[i])`.

### Examples
```

```

# Indexing Tuples with Tuples
Tuples can be indexed by tuples of `Integer`s and tuples of `Bool`s.  This is often more performant than indexing by vectors, ranges, or other iterators since the length (and sometimes even the value) of the resulting tuple is more likely to be inferred.

A caveat is that this functionality constitutes type piracy and is therefore **theoretically unsafe**: it could potentially cause unexpected behavior in unrelated code.  This possibility seems unlikely, however, as it could only occur in the case of external code that either (1) relies on tuples-indexed-by-tuples to be an error, or (2) itself implements tuples-indexed-by-tuples. 
```
getindex(t::Tuple, i::Tuple{Vararg{Integer}})
```
returns the tuple `(t[i[1]], t[i[2]], ...)`.

```
getindex(t::Tuple, m::Tuple{Vararg{Bool}})
```
returns a tuple whose elements are selected by the `true` elements of `m`.

```
setindex(t::Tuple, v, i::Dims)
```
Construct a tuple by effectively replacing selected elements of `t`.  The result `r` is equivalent to `r = t; r[i] = v`.

<!-- ```
deleteat(t::Tuple, inds)
```
Construct a tuple by effectively removing selected elements of `t`.
 -->

### Examples


# Computations on Tuples
```
cumsum(t::Tuple)
cumprod(t::Tuple)
```
These methods are faster than the generic implementations in Base.

```
cumfun(t::Tuple, op)
```
Return a tuple `c` generated by applying `op` cumulatively to the elements of `t`.  Specifically, `c[1] = t[1]` and `c[k] = op(c[k-1], t[k])` for k>1.

<!-- ### Examples


# Searching
Several methods are provided to  -->

# Sorting and Permutation
```
invperm(p::Tuple)
```
Return the inverse of a perumtation `p`.

```
sort(t::Tuple, ...)
sortperm(t::Tuple, ...)
```
Sort a tuple, or return a permutation that sorts a tuple. These are reimplementations of the methods in Base, with better performance.


# Type Pirated Methods

This package introduces the following methods which constitute type piracy:

| Method | Effect |
| --- | --- |
| `ntuple(f, n, ::Type)` | new behavior |
| `ntuple(f, Val(n), ::Type)` | new behavior |
| `getindex(::Tuple, ::NTuple{N,Int})` | new behavior |
| `getindex(::Tuple, ::NTuple{N,Bool})` | new behavior |
| `setindex(::Tuple, v, ::NTuple{N,Int})` | new behavior |
| `invperm(::Tuple)` | improved performance |
| `sort(::Tuple)` | improved performance |
| `sortperm(::Tuple)` | improved performance |
| `findfirst(::Tuple)` | improved performance |
| `cumsum(::Tuple)` | improved performance |
| `cumprod(::Tuple)` | improved performance |


# Under the Hood
Operations on small tuples (generally of length ≤ 10) are typically achieved via explicit formulas and/or compiler-inferrable recursive constructions.  Operations on larger `NTuple`s use `MVector`s as temporary fixed-size mutable spaces.
