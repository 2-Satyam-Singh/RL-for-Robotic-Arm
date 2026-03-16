import curses, time, random

def main(stdscr):
    curses.curs_set(0)
    rows, cols = stdscr.getmaxyx()
    grid = [[random.randint(0,1) for _ in range(cols)] for _ in range(rows)]

    while True:
        stdscr.clear()
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    if j < cols - 1 and i < rows - 1:  # avoid bottom-right crash
                        stdscr.addstr(i, j, "█")
        stdscr.refresh()

        new_grid = [[0]*cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                neighbors = sum(grid[x][y] 
                    for x in [(i-1)%rows,i,(i+1)%rows] 
                    for y in [(j-1)%cols,j,(j+1)%cols]
                    if not (x==i and y==j))
                if grid[i][j] == 1 and neighbors in [2,3]:
                    new_grid[i][j] = 1
                elif grid[i][j] == 0 and neighbors == 3:
                    new_grid[i][j] = 1
        grid = new_grid
        time.sleep(0.1)

curses.wrapper(main)
