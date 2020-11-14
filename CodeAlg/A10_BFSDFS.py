
'''
Given an m x n 2d grid map of '1's (land) and '0's (water), return the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

BFS:

DFS:

'''

# Reference solution
# BFS solution
def numIslands(self, grid: List[List[str]]) -> int:

    # method 1: brute force
    # edge island
    # center island
    # problem -- too low efficiency

    # method 2: DP
    # similar to DK algorithm
    # rule and idea: all the next steps are zeros

    # Stopping rules:
    # 1. all the next steps are zeros

    # DFS algorithms -- pending

    # Main function
    if not grid:
        return 0

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):

            if grid[i][j] == '1':
                # Start the searching from a node
                self.dfs(grid, i, j)
                count += 1

    return count


def dfs(self, grid, i, j):
    if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
        return

    # cover the visited note
    grid[i][j] = '#'

    self.dfs(grid, i + 1, j)
    self.dfs(grid, i - 1, j)
    self.dfs(grid, i, j + 1)
    self.dfs(grid, i, j - 1)

# BFS solution















