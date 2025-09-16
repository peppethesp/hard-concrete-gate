#  %%
# Python3 program to print all combination
# of size r in an array of size n

import numpy as np


def CombinationRepetitionUtil(chosen, arr, index, r, start, end, combination):
    """
        chosen[]: Temporary array to store current combination
        arr[]: Input Array of the elements whose combination has to be computed
        start & end: Starting and Ending indexes in arr[] 
        r: Size of a combination to be printed
    """
    # Current combination is ready,  print it
    if index == r:
        # for j in range(r):
            # print(chosen[j], end = " ")
        # print()
        combination.append(chosen.copy())
        return
        
    # When no more elements are there to put in chosen[]
    if start > end:
        return
        
    # Current is included, put next at next location
    chosen[index] = arr[start]
    
    # Current is excluded, replace it with next (Note that i+1 is passed, but index is not changed)
    CombinationRepetitionUtil(chosen, arr, index+1, r, start, end, combination)
    CombinationRepetitionUtil(chosen, arr, index, r, start+1, end, combination)


# The main function that returns all combinations of order/class 'r' in arr[] of size n.
# This function mainly uses CombinationRepetitionUtil()
def CombinationRepetition(arr, n, r):
    """
    Generates all combinations with repetition from a given list of elements,
    of a  specified maximum combination order.

    Parameters:
    - arr (list): The input list of elements to generate combinations from.
    - n (int): The number of distinct elements in 'arr'.
    - r (int): The number of elements allowed in each combination.

    Returns:
    - List of lists containing all possible combinations with repetitions
    for orders ranging from 1 to r_max.
    """
    # A temporary array to store all combination one by one
    chosen = [0] * r
    
    # An array that returns all the combination of the elements in arr of size n of order/class 'r'
    combination = []
    # Create combination list using temporary array 'chosen[]'
    CombinationRepetitionUtil(chosen, arr, 0, r, 0, n, combination)
    return combination


# Driver code
# Nonlinearity order
def CombinationRepetitionComplete(r_max, n):
    """
    Code to generate all the combinations with repetitions of 'n' elements up to order 'r_max'

    Parameters:
    - n (int): The number of distinct elements to choose from.
    - r_max (int): The maximum number of elements in each combination.

    Returns:
    - List of lists of lists representing all combinations with repetitions for orders 1 to r_max.
    combination_total[i]: list of lists with the combinations of order i+1
    combination_total[i][j]: list with the j-th combination (order i+1)
    """
    stateVar = range(n)
    combination_total = []

    for r in range(1, r_max+1):
        n = len(stateVar) - 1
        combination_total.append( CombinationRepetition(stateVar, n, r) )
    return combination_total
    

# %%
# Example of usage:

if __name__ == "__main__":
    combination_total = CombinationRepetitionComplete(3, 2)
    for i, elem in enumerate(combination_total):
        # loop through the polynomial nonlinearity order
        for j in range(len(combination_total[i])):
            print("term: ", end=" ")
            # loop through the various combinations 
            for k in range(len(combination_total[i][j])):
                print(combination_total[i][j][k], end=" ")
            print()

