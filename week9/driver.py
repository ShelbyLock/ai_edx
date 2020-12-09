#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:52:01 2020

@author: qianyingliao
"""

from collections import deque
import copy
import sys

dictKeys1 = ['A','B','C','D','E','F','G','H','I']
dictKeys2 = ['1','2','3','4','5','6','7','8','9']
domain = list(range(1,10,1))

#------------------------------------------------------------------------------
# Helper Methods
#------------------------------------------------------------------------------
def queryBoxKeys(row, col):
    boxRow = row // 3
    boxCol = col // 3
    boxKey = boxRow * 3 + boxCol
    return boxRow, boxCol, boxKey

def generateBoxNeighbors():
    boxesNeighbors = dict()
    for rowIndex, rowKey in enumerate(dictKeys1):
        for colIndex, colKey in enumerate(dictKeys2):
            boxRow = rowIndex // 3
            boxCol = colIndex // 3
            boxKey = boxRow * 3 + boxCol      
            if boxKey in boxesNeighbors:
                continue    
            boxesNeighbors[boxKey] = [key1+key2 for key1 in dictKeys1[boxRow*3: (boxRow+1)*3] for key2 in dictKeys2[boxCol*3: (boxCol+1)*3]]     
    return boxesNeighbors

def queryRowColumnNeighbors(row, col):
    rowKey = dictKeys1[row]
    colKey = dictKeys2[col]
    rowNeighbors = [rowKey+key for key in dictKeys2]
    colNeighbors = [key+colKey for key in dictKeys1 if key!=rowKey]
    return rowNeighbors+colNeighbors

def queryBoxNeighbors(row, col, boxesNeighbors):
    _,_,boxKey = queryBoxKeys(row, col)
    return boxesNeighbors[boxKey]

def queryNeighbors(row, col, boxesNeighbors):
    Neighbors_ColRow = queryRowColumnNeighbors(row, col)
    Neighbors_Box = queryBoxNeighbors(row, col, boxesNeighbors)
    result = Neighbors_ColRow + list(set(Neighbors_Box) - set(Neighbors_ColRow))
    return result

def print_sudoku(inp):
    print("+" + "---+"*9)
    for row in range(9):
        print(("|" + " {}   {}   {} |"*3).format(*[x if x != '0' else " " for x in inp[row*9:(row+1)*9]]))
        if row % 3 == 2:
            print("+" + "---+"*9)
        else:
            print("+" + "   +"*9)

def generateCSP(problem, boxesNeighbors):
    board = dict()
    neighbors = dict()
    for i, item in enumerate(problem):
        item = int(item)
        row = i // 9
        col = i % 9
        dictKey = dictKeys1[row] + dictKeys2[col]
        if item == 0:
            board[dictKey] = domain.copy()
        else:
            board[dictKey] = [item]
        temp = queryNeighbors(row, col, boxesNeighbors)
        temp.remove(dictKey)
        neighbors[dictKey] = temp
    return board, neighbors

#------------------------------------------------------------------------------
# AC-3 Methods
#------------------------------------------------------------------------------

def REVISE(board, X_i, X_j):
    revised = False
    temp = board[X_i].copy()
    for x in board[X_i]:
        sum_unequal = sum(1 for y in board[X_j] if x != y)
        if sum_unequal == 0:
            temp.remove(x)
            revised = True
    board[X_i] = temp.copy()
    return revised

def AC_3(board, neighbors):
    queue = deque()
    #Initialize Queue
    for key, key_neignbors in neighbors.items():
        queue.extend((key, neighbor)for neighbor in key_neignbors)
    while queue:
        X_i, X_j = queue.popleft()
        revised = REVISE(board, X_i, X_j)
        if revised:
            if not board[X_i]:
                return False
            queue.extend((X_k, X_i) for X_k in neighbors[X_i] if X_k!=X_j)
    return True

#------------------------------------------------------------------------------
# BTS Methods
#------------------------------------------------------------------------------

def BTS(board, neighbors):
    return BACKTRACK(dict(), board, neighbors)
    
def BACKTRACK(assignment, board, neighbors):

    if assignment.keys() == board.keys():
        return assignment
    key = MRV(board, assignment)
    values = LCV(key, board, neighbors)
    for value in values:
        if checkConsistency(key, value, assignment, neighbors):
            assignment[key] = value
            # Inference
            checked, domain = ForwardChecking(key, value, board, neighbors)
            result = BACKTRACK(assignment, domain, neighbors)
            if result is not False:
                return result      
            assignment.pop(key)

    #print(assignment)
    return False

def MRV(domain, assignment):
    minimum_len = 9
    selected_key = ""
    for key, value in domain.items():
        if key in assignment.keys():
            continue
        len_val = len(value)
        if len_val <= minimum_len:
            minimum_len = len_val
            selected_key = key
    return selected_key

def LCV(key, domain, neighbors):
    key_values = domain[key].copy()
    key_neighbors = neighbors[key]
    rule_out_counts = dict()
    if len(key_values) == 1:
        return key_values
    for val in key_values:
        rule_out_counts[val]=0
        for key_neighbor in key_neighbors:
            if val in domain[key_neighbor]:
                rule_out_counts[val]+=1
    rule_out_counts = {k: v for k, v in sorted(rule_out_counts.items(), key=lambda item: item[1])}
    return rule_out_counts.keys()

def checkConsistency(key, value, assignment, neighbors):
    key_neighbors = neighbors[key]
    violation_count = 0
    for neighbor in key_neighbors:
        if neighbor in assignment.keys():
            violation_count += (value == assignment[neighbor])
    return violation_count==0

#Propagate Information from assigned to unassigned variables
def ForwardChecking(key, value, board, neighbors):
    domain = copy.deepcopy(board)
    domain[key].remove(value)
    key_neighbors = neighbors[key]
    checked = False
    for key_neighbor in key_neighbors:
        neighbor_values = domain[key_neighbor].copy()
        if value in neighbor_values:
           domain[key_neighbor].remove(value)
           checked = True
    return checked, domain

def test(problem):
    boxesNeighbors = generateBoxNeighbors()
    board, neighbors = generateCSP(problem, boxesNeighbors)
    # AC-3
    status = AC_3(board, neighbors)  
    acum_count = sum(len(n) for n in board.values())
    actual_result = ""
    if status and (acum_count == len(problem)):
        number_string = ""
        for n in board.values():
            number_string+=str(n[-1])
        actual_result = str(number_string)+" AC3"        
        writeToFile(actual_result)       
        return actual_result

    # BTS
    status = BTS(board, neighbors)
    if status:
        #Organize anwer:
        number_string = ""
        for key in board.keys():
            number_string+=str(status[key])
        actual_result = str(number_string)+" BTS"        
        writeToFile(actual_result)        
        return actual_result
# -------------------------------------
## Main Function 
#1. reads in Input
#2. Runs corresponding Algorithm
# -------------------------------------
def writeToFile(actual_result):
    text_file = open("output.txt", "w")
    text_file.write(actual_result)
    text_file.close()

def main():
     problem = sys.argv[1] 
     test(problem)
     
if __name__ == '__main__':

    main()