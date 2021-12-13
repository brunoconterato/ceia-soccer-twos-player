## New experiment PPO With Reward Ray 1.4

**Utilizado na Competição!**

Para reproduzir o treinamento, execute o notebook experiment.ipynb

- Reward by ball velocity towards goal only last player that touched the ball;
- Only apply this reward for effective ball moviments (above a minimum value);
- Only considers ball velocity on the X axis;
- No punishment for player in front of ball;
- No League running here