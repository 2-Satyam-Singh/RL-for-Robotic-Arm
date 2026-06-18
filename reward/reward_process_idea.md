Reward step 1: First provide the rewards directly just if the object is touched.
How to identify: The location of object is moved in any random direction.

This will promote the robot to move towards to object.


Reward step 2: Provide rewards only when the object is pushed towards destination.

Ideas:
Put all three obstacles, but only give reward when robot pushes middle one.





I want to do this in v0.9:
1) bring a basket in the environment.
2) Both The objet and the basket shall start from random positions instead of fixed position, but the object should start at table and the basket should start within a fixed radius from the robot say 5 metres.
3) now the robot's task it to throw the object exactly in the basket.
I shall use a dense reward mechanism like, when the distance between robot's end effector and the object decreases, I shall give it a positive reward based on the distance, and then, once the robot hits the object, after that, no reward based on distance between end effector and object, instead reward based on distance between object and the basket.. say for example the basket is the goalpost for the football or the hole where we put the golf ball.