# 3 partition problem
## Problem
https://en.wikipedia.org/wiki/3-partition_problem

The 3-partition problem is a strongly NP-complete problem in computer science. The problem is to decide whether a given multiset of integers can be partitioned into triplets that all have the same sum.

#### Examples
 - ``` {1, 2, 3, 4, 5, 7} -> {1, 7, 3} {2, 4, 5} m=2 T=11 ```
 - ``` {4, 5, 5, 5, 5, 6} -> {4, 5, 6} {5, 5, 5} m=2 T=15 ```


## Project

-----------------------------------------------------------------
#### Goal function
The project solves the problem of solving algorithms in search of solutions.
The goal function serves as the basis for evaluating the solution
- ``` goal function returns real numbers from 0 to 1 ```
- ``` 0 = worst | 1 = solved | 0 < x < 1 - partially correct ```
-----------------------------------------------------------------
