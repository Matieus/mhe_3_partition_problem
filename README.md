# 3 partition problem
## Problem unrestricted-input variant
https://en.wikipedia.org/wiki/3-partition_problem

The 3-partition problem is a strongly NP-complete problem in computer science. The problem is to decide whether a given multiset of integers can be partitioned into triplets that all have the same sum.

#### Conditions
- ```- The set containing n positive integer elements.```
- ``` The set must be partitionable into m triplets, S1, S2, …, Sm, where n = 3m ```

#### Values

- ``` m - amout of triplets ```
- ``` T - target vaule ```  sum of all elements in the set divided by m


#### Examples
 - ``` {1, 2, 3, 4, 5, 7} -> {1, 7, 3} {2, 4, 5} m=2 T=11 ```
 - ``` {4, 5, 5, 5, 5, 6} -> {4, 5, 6} {5, 5, 5} m=2 T=15 ```
 - ``` {1, 2, 3, 3, 8, 9, 9, 9, 16} -> {2, 9, 9} {1, 3, 16} {8, 9, 3} m=3 T=20 ```

## Project

-----------------------------------------------------------------
#### Goal function
The project solves the problem of solving algorithms in search of solutions.
The goal function serves as the basis for evaluating the solution
- ``` goal function returns real numbers from 0 to 1 ```
- ``` 0 = worst | 1 = solved | 0 < x < 1 - partially correct ```


- If the sum of the triplet equals the target value ``` t ```
- for a given triplet gives 1, 
- otherwise it will count how far this sum is from the expected value ``` 0.5 / abs(self.p.t - x) ```
- In order not to award too high a score for a given triplet, the basis is 0.5 points if the absolute value of t is 1
- the higher the absolute value, the lower the triplet score
- then the ratings of all triplets are summed up and and divided by ``` m ```


-----------------------------------------------------------------
#### Neighbours
Neighbors are the set of ``` closest solutions ``` such that in each of them there is only one exchange of two elements
Not all possible ``` closest solutions ``` are in the set of neighbors.

For example:
- problem: ``` {1, 2, 3, 4, 5, 7, 8, 2, 1} t=11, m=2```
- neighbours ``` [[1, 5, 3, 4, 7, 2], [4, 5, 3, 1, 7, 2], [7, 5, 3, 4, 1, 2], [2, 5, 3, 4, 7, 1], [1, 4, 3, 5, 7, 2], [1, 7, 3, 4, 5, 2], [1, 2, 3, 4, 7, 5], [1, 5, 4, 3, 7, 2], [1, 5, 7, 4, 3, 2], [1, 5, 2, 4, 7, 3]] ```

The generate_neighbours function skips neighbors that exchange elements within only one triplet
Because such solutions give exactly the same result of the goal function.