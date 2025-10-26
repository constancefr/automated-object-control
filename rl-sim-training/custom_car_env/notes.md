# Car Simulator Design
- Skill: controlling an ego car to avoid collision with other cars
- Information: ego + other cars' current position, velocity and acceleration
- Actions: accelerate or brake by some fixed amount, or idle
- Success: avoid crashes!
- End: some arbitrary time limit (we want collision-freedom on infinite time horizon)

## Requirements from the model:
- ego car must observe the env at least every T seconds
- maintain a distance of at least L between cars to avoid collision
- velocity must be positive
- acceleration is bounded by B_max and A_max
- mimic the hybrid system model
    - continuous dynamics (ODEs)
    - discrete action space mapped to allowed acceleration range
- acceleration correction (if car tries to go backwards or exceed the speed limit)
- invariant state space (make sure the initial conditions sampled satisfy the invariant defined in the paper, pg. 12)
- ensure non-deterministic choices for the other car(s) within physical limitations
    - simulate sufficient worst-case scenarios! (emergency brakes)

## Addressing the Model2Sim gap:
- make sure the environment does NOT assume maximal braking power (dL conservatively proves safety based on min braking power)
- ensure that breaking always lead to deceleration (rather than acceleration - something to do with how velocity is changed, see pg. 13)
- include emergency braking scenarios (unlike what IDM does)
- proper sampling of initial conditions
- track whether or not we experience Euleru crashes (resolved with a smaller step size)

## TODO
- [X] make cars smaller
- [X] currently, the other car is fixed -> make it take random (?) CORRECT actions
- [ ] make sure the env/sim respects the specs above
- [ ] change up the visuals?
- [ ] connect a controller to the ego car -> RL training
- [ ] add a second other car behind
- [ ] train an agent to generate data
