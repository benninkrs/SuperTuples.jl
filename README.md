# SuperTuples.jl
Utilities for working with tuples in Julia

**SuperTuples** provides fast, convenient methods for working with tuples:

* Constructing
* Splitting and concatenating
* Indexing via other tuples
* Permuting
* Searching
* Sorting

Most of the new functionality is achieved by adding methods to standard Base functions (such as `getindex`). Since this constitutes [type piracy](https://docs.julialang.org/en/v1/manual/style-guide/#Avoid-type-piracy-1) there is the possibility that **using SuperTuples may lead to unexpected behavior in unrelated code (including Base)**, though I am not aware of any such behavior.

SuperTuples provides some similar functionality as, but was developed mostly independently of, [Tuple Tools](https://github.com/Jutho/TupleTools.jl).

# Functions

## Tuple Construction

## Indexing

## Permuting

## Searching

## Sorting

# Under the Hood
Operations on small tuples (generally of length <10) are typically achieved via explicit formulas and/or compiler-inferrable recursive constructions.  Operations on large `NTuple`s use `MVector`s as temporary fixed-size mutable spaces.
