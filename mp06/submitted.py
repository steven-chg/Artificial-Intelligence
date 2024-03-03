# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

import queue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function

    """
    Possible Maze API Calls
    1) maze.navigable(i, j) (to verify whether to add neighboring cells into the queue; don't add wall cells)
    2) maze.legend.waypoint (to verify whether we have reached the end point)
    3) maze.start (to get the starting point)
    4) maze.neighbors_all(i, j) (returns a tuple of sequence of all navigable neighbors of cell (i, j))
    """

    # BFS Implementation Using FIFO Queue

    # declare a 2D array to store the previous node of each node visited 
    parent_info = [[(-1, -1) for col in range(maze.size.x)] for row in range(maze.size.y)]
    parent_info[maze.start[0]][maze.start[1]] = (-2, -2)

    q = queue.Queue()
    q.put(maze.start)
    visited = set()
    maxX = maze.size.x - 1
    maxY = maze.size.y - 1

    # top, topright, right, bottomright, bottom, bottomleft, left, topleft
    # directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
 
    # BFS traverse until we reach the waypoint
    while not q.empty():
        currentCell = q.get()
        # we need to account for the possibility that multiple instances of the same cell are on the queue
        while(currentCell in visited):
            currentCell = q.get()
        visited.add(currentCell)

        # if we have reached the waypoint, then break out of while loop
        if maze[currentCell] == maze.legend.waypoint:
            break

        # USING MAZE.NEIGHBORS_ALL
        # add all neighbors (that are navigable) to the queue
        for newCell in maze.neighbors_all(currentCell[0], currentCell[1]):
            newRow = newCell[0]
            newCol = newCell[1]
            # condition checks (boundary, visited)
            if newRow >= 0 and newRow <= maxY and newCol >= 0 and newCol <= maxX and newCell not in visited:
                # add to queue
                q.put(newCell)
                # set up parent information (ONLY IF IT HASN"T ALREADY BEEN SET UP YET)
                if parent_info[newCell[0]][newCell[1]] == (-1, -1):
                    parent_info[newCell[0]][newCell[1]] = currentCell

        # # USING DIRECTIONS
        # # add all neighbors (that are navigable) to the queue
        # for changeRow, changeY in directions:
        #     newRow, newCol = currentCell[0] + changeRow, currentCell[1] + changeY
        #     newCell = (newRow, newCol)
        #     # condition checks (boundary, navigable, visited)
        #     if newRow >= 0 and newRow <= maxY and newCol >= 0 and newCol <= maxX and maze.navigable(newRow, newCol) and newCell not in visited:
        #         # add to queue
        #         q.put(newCell)
        #         # set up parent information (ONLY IF IT HASN"T ALREADY BEEN SET UP YET)
        #         if parent_info[newCell[0]][newCell[1]] == (-1, -1):
        #             parent_info[newCell[0]][newCell[1]] = currentCell
    
    # Find the path using parent_info 
    path = []
    currentPoint = maze.waypoints[0]
    # loop until we reach the start point, which has parent info (-2, -2)
    while parent_info[currentPoint[0]][currentPoint[1]] != (-2, -2):
        path.insert(0, currentPoint)
        currentPoint = parent_info[currentPoint[0]][currentPoint[1]]
    path.insert(0, maze.start)

    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single

    wayPoint = maze.waypoints[0]

    # Heuristic 
    # max(abs(currentCell[0] - wayPoint[0]), abs(currentCell[1] - wayPoint[1]))




    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
