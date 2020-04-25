import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId=p.loadURDF("plane.urdf")
cubeStartPos=[0,0,1]
cubeStartOrientation=p.getQuaternionFromEuler([0,0,0])
boxId=p.loadURDF("r2d2.urdf",cubeStartPos,cubeStartOrientation)

for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()