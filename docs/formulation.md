# Problem formulation

The choices I had to make because the task does not specify them, and why I made them. I am
writing this as I go rather than at the end, and I kept the options I rejected, because that
is usually where the reasoning is.

## What the task fixes

These came with the task. I did not choose them, and I have not tuned them, which is what
makes testing the rewards later worth running.


|              |                                                                                               |
| ------------ | --------------------------------------------------------------------------------------------- |
| grid         | 8 × 8, robot starts at (0,0), goal at (7,7)                                                   |
| cow          | starts at the center, and each step moves one cell in a random direction with probability 0.7 |
| episode ends | robot reaches the goal, robot enters the cow's cell, or 60 steps elapse                       |
| too close    | the cow is in one of the 8 cells surrounding the robot                                        |



| event                                  | reward |
| -------------------------------------- | ------ |
| reach the goal                         | +20    |
| collision with the cow                 | −10    |
| too close, charged every step it holds | −3     |
| every step                             | −0.1   |


## 1. State

### What I chose

`(robot_row, robot_col, dr, dc)`, where `dr` and `dc` are the cow's position minus the
robot's, each clipped to the range −2 to +2.

That is 8 × 8 × 5 × 5 = **1,600 states**, or 8,000 Q-values once I add the 5 actions.

### Where the clipping lives, and why I moved it

My first version had the environment do the clipping. `CowWorld(clip=2)` took the radius as a
constructor argument and `reset()` handed back the packed integer directly, so the observation
space was `Discrete(1600)`.

I changed it after reading the Gymnasium custom-environment tutorial the task links to, which
reports structured coordinates and stops there. Sutton & Barto draw the same line from the
theory end: the Markov property is "best viewed as a restriction not on the decision process,
but on the state" (sec 3.1, p.49). The world has exact positions. Deciding that two distant
cows count as the same situation is something I am doing to keep a table small, so it should
sit with the learner.

I had already run into this without noticing. My `_info()` dict was handing true positions
back out, because the rule-based baseline could not work from an observation I had compressed.
Needing a way around my own observation should have told me something.

The environment now reports `MultiDiscrete([8, 8, 8, 8])` — robot row, robot col, cow row, cow
col — and `encoding.py` turns that into a table index at whatever radius the agent asks for.
That gets me ±2 and ±3 running against one environment instance on identical seeds, rather
than two environments I would have to trust were set up the same way, and it puts the baseline
back on the same observation as everything else. It also leaves the door open to DQN or PPO,
which the task allows; `Discrete(1600)` would have meant one-hot vectors of length 1600.

Everything the tutorial lists as required is unchanged: subclassing `gym.Env`, declaring both
spaces, `reset(seed, options)` returning `(obs, info)` with `super().reset(seed=seed)` called
first, and `step` returning the five-tuple.

### The goal is not in the state

I first thought it should be. It should not. The goal is fixed at (7,7), so it holds the same
value in every state and carries no information at all. The robot's own position already
covers it: if the robot is at (2,3), the goal is 9 steps away and down-right. If the goal
moved between episodes this would be a different decision.

### Why the cow is an offset, and why it is clipped

The offset is `cow − robot`, one number per axis. Both coordinates live in 0–7, so the
offset runs from `0 − 7 = −7` up to `7 − 0 = +7`. That is 15 values per axis, not 16: eight
cells means the largest gap between two of them is seven, the same way a ruler marked 0 to 7
has a longest span of 7. Clipping to ±2 leaves 5 of those 15, and everything past the edge
folds onto it.

![the offset range before and after clipping](figures/offset_range.png)

Using the cow's position relative to the robot was my first instinct, and on its own it is
worse, not better. 15 × 15 = 225 combinations per cow, and I still need the robot's own
position to navigate, so 64 × 225 = 14,400 states — 3.5x worse than just storing both
absolute positions. Clipping is what makes the relative version pay.

How far is far enough? The robot moves one cell and the cow moves one cell, so in one step
they can close two cells of distance:

- At distance 2 the cow can reach the robot's cell next step. That is a collision: −10, and
the episode is over.
- At distance 3 the worst case is ending up adjacent, which costs the −3 too-close penalty
but leaves the robot alive and moving.

![why the radius is two](figures/why_radius_two.png)

So **±2 is the smallest radius where every cow that can hit me next step is still visible**.
I did not pick 2 because it looked reasonable. I picked it because that is where the one-step
reachability line falls, and ±1 would leave the agent blind to a cow that can kill the
episode.

![what clipping merges](figures/clip_merges_states.png)

What the clip buys is that states which should behave the same stop being separate. With the
robot at (3,3), a cow at (3,5), (3,6) or (3,7) all become the same row of the table. None of them
can reach the robot next step, so the best action is the same in all three. Without clipping
the agent has to learn that lesson three separate times. For a tabular method this matters:
there is no network to generalize across similar states, so every state is learned on its own
visits.

Sutton & Barto are explicit that convergence needs every state-action pair to keep being
updated -- "all that is required for correct convergence is that all pairs continue to be
updated" (sec 6.5, p.131). Under a fixed episode budget that turns into arithmetic: fewer rows
means more visits per row, and 4,096 rows would spread the same experience over two and a half
times as many of them.

### What it costs

Past the radius the agent cannot tell a cow moving toward it from one moving away. I accepted
that because the cow moves uniformly at random. There is no drift to detect, so the
information is not predictive. However, this stops being true in the harder world where the cow moves toward the robot, and I will have to revisit the radius there.

### Rejected


| alternative                               | states | why not                                                    |
| ----------------------------------------- | ------ | ---------------------------------------------------------- |
| absolute positions `(r_r, r_c, c_r, c_c)` | 4,096  | learns the exact position of a far cow as a separate case  |
| relative offset, not clipped              | 14,400 | loses the robot's own position and is 3.5x larger          |
| goal position included                    | —      | fixed at (7,7), so it is a constant                        |
| clipped at ±1                             | 576    | a cow at distance 2 is invisible and can collide next step |


### Still open: ±3

I assumed a wider radius would be safer. It might not be. With a fixed number of training
episodes, 3,136 states means each state is visited about half as often and the Q-values come
out noisier. More sight is not automatically more safety if the data budget does not grow with
it.

I am building ±2 first because it is the radius I can justify from the dynamics, then
measuring ±3 on the same seeds. Whichever wins, the comparison is the result. I would rather
measure this than argue about it.

### Only 1,156 of the 1,600 rows are reachable

Not every index the encoding can produce corresponds to a situation the grid can actually
produce. The largest index would need the robot at (7,7) and the cow two rows below the
bottom of the board. Checking this exhaustively, 1,156 of the 1,600 rows are addressable and
the mapping onto them is one-to-one. The unreachable rows stay at zero and cost nothing.

### Is the state Markov?

I went and read up on this because I was about to throw information away and wanted to know
whether I was allowed to. The task points at Sutton & Barto chapter 6 for the tabular
Q-learning update; the Markov property itself is defined back in 3.1, and it is the assumption
sitting underneath that update.

Their wording is that the probability of the next state and reward "depends on the immediately
preceding state and action [...] and, given them, not at all on earlier states and actions",
and that the state "must include information about all aspects of the past agent–environment
interaction that make a difference for the future" (p.49).

The reason it matters here is mechanical rather than philosophical: `Q(s, a)` is one number. If
two different pasts land on the same `s` and then behave differently, that single number has to
serve both and settles somewhere in between.

The full `(robot, cow)` state looks fine to me. The cow draws a fresh direction every step and
keeps nothing from the last one, so two cows on the same cell are in the same situation no
matter how they arrived.

![memoryless cow vs a cow with momentum](figures/markov_memoryless.png)

The version that would break it is a cow with momentum. If it tended to keep going the way it
was already heading, its position would not be enough — I would need its last direction too,
and that is not in the state. Worth remembering if the harder world later gives the cow a
preferred direction.

### Where clipping breaks it

Clipping blurs the photograph past the edge of the radius. With the robot at (3,3), a cow at
(3,5) and a cow at (3,7) encode to the same state, but their futures are not the same: the
first can be adjacent next step, the second cannot.

![two cows the agent cannot tell apart](figures/markov_clip_blur.png)

So the agent cannot separate two situations whose futures are not the same, which is the thing
the property rules out. Strictly that makes this a POMDP and not a clean MDP, and I only
realized it after I had already settled on the radius.

I am keeping it anyway. The situations being merged mostly want the same action, which was the
reason for merging them in the first place, so the shared Q-value averages over cases that
agree and the averaging costs little.

What I like about this one is that it does not have to stay an argument. If the blur is really
costing the policy, ±3 should come out ahead of ±2, since a wider radius blurs less. That is
now a second reason to run that comparison.

### Measuring "too close"

The task defines too close as the cow being in one of the 8 cells around the robot, so the
distance that matches it is Chebyshev -- max of the two axis gaps, the way a king moves in
chess. Chebyshev distance 1 is those 8 cells and nothing else.

Manhattan distance does not work here. It adds the two gaps instead of taking the larger one,
so the four diagonal neighbors come out at distance 2 and would escape the penalty, while the
four orthogonal ones come out at 1. The cow would be charged for standing beside the robot but
not for standing at its corner, which is not what the task describes and not how a real robot
would think about clearance.

| neighbor | Manhattan | Chebyshev |
|---|---|---|
| (3,4) from (3,3) | 1 | 1 |
| (4,4) from (3,3) | **2** | 1 |

I use Manhattan in one other place, as the rule-based baseline's distance to the goal, and it
is correct there for the opposite reason: the robot moves on 4 directions and never
diagonally, so the number of steps it needs really is the sum of the two gaps.


## 2. Actions

Five actions (up, down, left, right, stay) taken from the task as given. The index into
`MOVES` is the action number.

I kept `stay` rather than dropping it, because it is the only action that lets the robot wait
for the cow to wander off instead of committing to a detour. Whether the learned policy
actually uses it is something I want to read off the results.

## 3. Reward

The four numbers are fixed by the task and listed at the top. There is nothing for me to
decide here, so the open question is not what to set them to but what they will do to the
policy.

### A prediction, written before I run anything

Chapter 6 has a worked example close to this problem. In Cliff Walking (Example 6.6, p.132) a
gridworld has a strip that costs −100 and resets the episode. Q-learning learns the optimal
path, which runs right along the edge of it, while Sarsa learns a longer route further away.
Q-learning then does worse in practice, because ε-greedy exploration keeps pushing it over the
edge even though the values it learned are the optimal ones.

Our version differs in two ways I think matter. The cow wanders, so there is no stable edge to
hug the way there is with a static cliff. And the −3 is charged every step the robot stands
next to the cow rather than once on contact, so three steps spent adjacent costs more than
walking straight into it. That is an odd shape for a reward — it comes from the task, I did
not pick it — but it should make distance worth paying for.

So I expect the learned policy to land closer to the Sarsa side of that example than the
Q-learning side: longer routes, fewer collisions, even though I am running Q-learning. If it
instead learns to skim past the cow, I have read the reward wrong.

I am evaluating greedily with exploration off, so the particular failure in Cliff Walking —
good values, bad behavior because ε keeps acting — should not reach the results table. It
would show up in the training curve.

## 4. Step order

*Not decided yet. The task says the episode ends when the robot "enters the cow's cell" and
does not say what happens when the cow walks into the robot. Those are different events and I
have to pick one.*

## 5. Evaluation

*Not decided yet.*

## References

Sutton, R. S. and Barto, A. G., *Reinforcement Learning: An Introduction*, 2nd edition, 2020.
The task lists chapter 6 as background reading.

- sec 3.1, p.49 — the Markov property, and the point that it restricts the state rather than
  the process
- sec 6.5, p.131 — the Q-learning update (6.8), and the convergence requirement that all
  state-action pairs keep being updated
- Example 6.6, p.132 — Cliff Walking, where Q-learning takes the optimal risky path and Sarsa
  takes the safer one

Gymnasium, *Create a Custom Environment* — the required structure for `gym.Env`, which this
environment follows apart from reporting coordinates instead of a packed index.
