

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 20  # grid height
MAZE_W = 20  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 15, UNIT*14])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT*16, UNIT * 14])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        hell3_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
        hell4_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')
        hell5_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')
        hell6_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')
        hell7_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')
        hell8_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - 15, hell8_center[1] - 15,
            hell8_center[0] + 15, hell8_center[1] + 15,
            fill='black')
        hell9_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell9 = self.canvas.create_rectangle(
            hell9_center[0] - 15, hell9_center[1] - 15,
            hell9_center[0] + 15, hell9_center[1] + 15,
            fill='black')
        hell10_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell10 = self.canvas.create_rectangle(
            hell10_center[0] - 15, hell10_center[1] - 15,
            hell10_center[0] + 15, hell10_center[1] + 15,
            fill='black')
        hell11_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell11 = self.canvas.create_rectangle(
            hell11_center[0] - 15, hell11_center[1] - 15,
            hell11_center[0] + 15, hell11_center[1] + 15,
            fill='black')
        hell12_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell12 = self.canvas.create_rectangle(
            hell12_center[0] - 15, hell12_center[1] - 15,
            hell12_center[0] + 15, hell12_center[1] + 15,
            fill='black')
        hell13_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell13 = self.canvas.create_rectangle(
            hell13_center[0] - 15, hell13_center[1] - 15,
            hell13_center[0] + 15, hell13_center[1] + 15,
            fill='black')
        hell14_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell14 = self.canvas.create_rectangle(
            hell14_center[0] - 15, hell14_center[1] - 15,
            hell14_center[0] + 15, hell14_center[1] + 15,
            fill='black')
        hell15_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell15 = self.canvas.create_rectangle(
            hell15_center[0] - 15, hell15_center[1] - 15,
            hell15_center[0] + 15, hell15_center[1] + 15,
            fill='black')
        hell16_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell16 = self.canvas.create_rectangle(
            hell16_center[0] - 15, hell16_center[1] - 15,
            hell16_center[0] + 15, hell16_center[1] + 15,
            fill='black')
        hell17_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell17 = self.canvas.create_rectangle(
            hell17_center[0] - 15, hell17_center[1] - 15,
            hell17_center[0] + 15, hell17_center[1] + 15,
            fill='black')
        hell18_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell18 = self.canvas.create_rectangle(
            hell18_center[0] - 15, hell18_center[1] - 15,
            hell18_center[0] + 15, hell18_center[1] + 15,
            fill='black')
        hell19_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell19 = self.canvas.create_rectangle(
            hell19_center[0] - 15, hell19_center[1] - 15,
            hell19_center[0] + 15, hell19_center[1] + 15,
            fill='black')
        hell20_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell20 = self.canvas.create_rectangle(
            hell20_center[0] - 15, hell20_center[1] - 15,
            hell20_center[0] + 15, hell20_center[1] + 15,
            fill='black')
        hell21_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell21 = self.canvas.create_rectangle(
            hell21_center[0] - 15, hell21_center[1] - 15,
            hell21_center[0] + 15, hell21_center[1] + 15,
            fill='black')
        hell22_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell22 = self.canvas.create_rectangle(
            hell22_center[0] - 15, hell22_center[1] - 15,
            hell22_center[0] + 15, hell22_center[1] + 15,
            fill='black')
        hell23_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell23 = self.canvas.create_rectangle(
            hell23_center[0] - 15, hell23_center[1] - 15,
            hell23_center[0] + 15, hell23_center[1] + 15,
            fill='black')          
        hell24_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell24 = self.canvas.create_rectangle(
            hell24_center[0] - 15, hell24_center[1] - 15,
            hell24_center[0] + 15, hell24_center[1] + 15,
            fill='black')          
        hell25_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell25 = self.canvas.create_rectangle(
            hell25_center[0] - 15, hell25_center[1] - 15,
            hell25_center[0] + 15, hell25_center[1] + 15,
            fill='black')          
        hell26_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell26 = self.canvas.create_rectangle(
            hell26_center[0] - 15, hell26_center[1] - 15,
            hell26_center[0] + 15, hell26_center[1] + 15,
            fill='black')          
        hell27_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell27 = self.canvas.create_rectangle(
            hell27_center[0] - 15, hell27_center[1] - 15,
            hell27_center[0] + 15, hell27_center[1] + 15,
            fill='black')          
        hell28_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell28 = self.canvas.create_rectangle(
            hell28_center[0] - 15, hell28_center[1] - 15,
            hell28_center[0] + 15, hell28_center[1] + 15,
            fill='black')          
        hell29_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell29 = self.canvas.create_rectangle(
            hell29_center[0] - 15, hell29_center[1] - 15,
            hell29_center[0] + 15, hell29_center[1] + 15,
            fill='black')          
        hell31_center = origin + np.array([UNIT*np.random.randint(1,20), UNIT * np.random.randint(1,20)])
        self.hell31 = self.canvas.create_rectangle(
            hell31_center[0] - 15, hell31_center[1] - 15,
            hell31_center[0] + 15, hell31_center[1] + 15,
            fill='black')          

        # create oval
        oval_center = origin + UNIT * 15
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1),
                    self.canvas.coords(self.hell2),
                    self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4),
                    self.canvas.coords(self.hell5),
                    self.canvas.coords(self.hell6),
                    self.canvas.coords(self.hell7),
                    self.canvas.coords(self.hell8),
                    self.canvas.coords(self.hell9),
                    self.canvas.coords(self.hell10),
                    self.canvas.coords(self.hell11),
                    self.canvas.coords(self.hell12),
                    self.canvas.coords(self.hell13),
                    self.canvas.coords(self.hell14),
                    self.canvas.coords(self.hell15),
                    self.canvas.coords(self.hell16),
                    self.canvas.coords(self.hell17),
                    self.canvas.coords(self.hell18),
                    self.canvas.coords(self.hell19),
                    self.canvas.coords(self.hell20),
                    self.canvas.coords(self.hell21),
                    self.canvas.coords(self.hell22),
                    self.canvas.coords(self.hell23),
                    self.canvas.coords(self.hell24),
                    self.canvas.coords(self.hell25),
                    self.canvas.coords(self.hell26),
                    self.canvas.coords(self.hell27),
                    self.canvas.coords(self.hell28),
                    self.canvas.coords(self.hell29),
                    self.canvas.coords(self.hell31)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.01)
        self.update()